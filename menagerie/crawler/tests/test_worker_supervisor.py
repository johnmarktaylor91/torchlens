"""Tests for parent-owned subprocess isolation and timeouts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pytest

import menagerie.crawler.worker_supervisor as worker_supervisor_module
from menagerie.crawler.authority import WorkerLease
from menagerie.crawler.driver import (
    _collect_worker_executable_closure,
    _compile_worker_read_manifest,
    _execution_identity,
)
from menagerie.crawler.identity import compute_recipe_revision, hash_bytes, stable_hash
from menagerie.crawler.proposal import model_code_manifest
from menagerie.crawler.tests.conftest import (
    RealEnvironmentFixture,
    make_author_proposal,
    make_proposed_artifact,
)
from menagerie.crawler.worker_supervisor import (
    clear_worker_lease,
    current_boot_id,
    open_worker_lease,
    process_start_token,
    reconcile_worker_lease,
    run_isolated_subprocess,
    supervise_worker,
)

import shutil
from types import ModuleType
from menagerie.crawler.constants import RunMode
from menagerie.crawler.proposal import ProposalValidationError, validate_author_proposal
from menagerie.crawler.recipe import RecipeError, load_declarative_recipe
from menagerie.crawler.standard_inputs import InputSpec
from menagerie.crawler.tests.test_slice_d_proposal_author import _ground_proposal
from menagerie.crawler.worker import WorkerRequest, run_worker
from menagerie.crawler.worker_supervisor import (
    _macos_denial_audit,
    _parent_owned_audit_path,
    _parse_linux_denial_audit,
)
from menagerie.crawler.policy import detect_os_sandbox
from menagerie.crawler.tests.conftest import make_worker_result_v3_mapping
from menagerie.crawler.worker_supervisor import (
    _MACOS_AUDIT_COMPLETION_MARKER,
    _parse_macos_denial_audit,
    poison_receipt_for_sandbox_denial,
)


def _timestamp_after(seconds: float) -> str:
    """Return a canonical future UTC timestamp.

    Parameters
    ----------
    seconds:
        Offset from the current wall clock.

    Returns
    -------
    str
        RFC 3339 timestamp with a ``Z`` suffix.
    """

    return (
        (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")
    )


def _test_lease(tmp_path: Path, *, lease_id: str = "lease-held") -> WorkerLease:
    """Build one current-boot pre-spawn lease for recovery attacks.

    Parameters
    ----------
    tmp_path:
        Isolated lease root.
    lease_id:
        Immutable synthetic lease identity.

    Returns
    -------
    WorkerLease
        Valid pre-spawn lease owned by this test process.
    """

    driver_token = process_start_token(os.getpid())
    assert driver_token is not None
    return WorkerLease(
        lease_id=lease_id,
        nonce=f"nonce-{lease_id}",
        run_id="run-held",
        stable_id="m_held",
        work_id="work-held",
        request_identity="sha256:" + "1" * 64,
        execution_identity="sha256:" + "2" * 64,
        boot_id=current_boot_id(),
        driver_pid=os.getpid(),
        driver_start_token=driver_token,
        child_pid=None,
        child_start_token=None,
        child_pgid=None,
        receipt_path=tmp_path / "receipt.json",
        opened_at=_timestamp_after(-1),
        deadline_at=_timestamp_after(30),
    )


@contextmanager
def _detached_lock_holder(
    tmp_path: Path,
) -> Iterator[tuple[subprocess.Popen[bytes], WorkerLease, Path, Path]]:
    """Yield a detached child that exclusively holds a production worker lease.

    Parameters
    ----------
    tmp_path:
        Isolated lock and record root.

    Yields
    ------
    tuple[subprocess.Popen[bytes], WorkerLease, Path, Path]
        Child, started lease, kernel lock path, and durable record path.
    """

    lock_path = tmp_path / "locks" / "worker.lock"
    record_path = tmp_path / "locks" / "worker-lease.json"
    handle = open_worker_lease(lock_path, record_path, _test_lease(tmp_path))
    assert handle.lock_fd is not None
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        pass_fds=(handle.lock_fd,),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    token = process_start_token(child.pid)
    deadline = time.monotonic() + 2
    while token is None and time.monotonic() < deadline:
        time.sleep(0.01)
        token = process_start_token(child.pid)
    assert token is not None
    started = replace(
        handle.lease,
        child_pid=child.pid,
        child_start_token=token,
        child_pgid=child.pid,
    )
    worker_supervisor_module._atomic_write_worker_lease(record_path, started)
    for attribute in ("lifecycle_read_fd", "lifecycle_write_fd", "lock_fd"):
        descriptor = getattr(handle, attribute)
        if descriptor is not None:
            os.close(descriptor)
            setattr(handle, attribute, None)
    try:
        yield child, started, lock_path, record_path
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        record_path.unlink(missing_ok=True)


def test_supervisor_scrubs_credentials_and_enforces_timeout(tmp_path: Path) -> None:
    """A fresh argv-only child cannot see a secret and is killed at its wall cap."""

    observation = run_isolated_subprocess(
        [
            sys.executable,
            "-c",
            "import os,time; print(os.getenv('CRAWLER_SECRET_TOKEN')); time.sleep(2)",
        ],
        tmp_path / "supervisor",
        timeout_seconds=0.1,
        rss_limit_bytes=1024**3,
        base_environment={
            "PATH": "/usr/bin:/bin",
            "CRAWLER_SECRET_TOKEN": "never",  # pragma: allowlist secret
        },
    )

    assert observation.timed_out is True
    assert observation.signal_number == 9
    assert "never" not in observation.stdout_tail


def test_supervisor_rejects_atomic_flat_v1_worker_receipt(tmp_path: Path) -> None:
    """A legacy request cannot make its atomic flat-v1 receipt live authority."""

    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """from __future__ import annotations
import torch

class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)

def build_model() -> object:
    return Tiny()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    return ((torch.zeros(1, 4, device=device),), {})
""",
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"
    receipt = scratch / "result" / "receipt.json"
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "stable_id": "m_supervised",
                "recipe": {
                    "kind": "typed-adapter",
                    "path": str(adapter),
                    "adapter_sha256": hash_bytes(adapter.read_bytes()),
                },
                "modality": "unknown",
                "input_spec": {"shape": [1, 4], "dtype": "float32"},
                "scratch_root": str(scratch),
                "meaningful_modes": ["train", "eval"],
                "recipe_revision": compute_recipe_revision(
                    {"recipe_type": "typed-adapter", "path": adapter.name},
                    "unbound",
                    adapter_bytes=adapter.read_bytes(),
                ),
            }
        ),
        encoding="utf-8",
    )

    result = supervise_worker(
        request,
        receipt,
        scratch,
        timeout_seconds=20,
        rss_limit_bytes=12 * 1024**3,
    )

    assert result.observation.exit_code == 0
    assert result.receipt_error == "invalid-receipt:worker-result-envelope"
    assert result.worker_receipt is None


def test_v3_worker_binds_raw_receipt_attestation_manifest_and_child_lease(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """One v3 success requires all four frozen security associations."""

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    adapter = model_dir / "adapter.py"
    adapter.write_text(
        """from __future__ import annotations
import torch

class Tiny(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1

def build_model() -> object:
    return Tiny()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 2, device=device),), {})
""",
        encoding="utf-8",
    )
    adapter_digest = hash_bytes(adapter.read_bytes())
    code_manifest = [dict(row) for row in model_code_manifest(adapter, model_dir)]
    code_identity = stable_hash(code_manifest)
    proposal = make_author_proposal("m_v3_supervised")
    proposal["work_id"] = "work-v3-supervised"
    facts = proposal["proposed_facts"]
    facts["implementation"].update(
        {
            "recipe_type": "typed-adapter",
            "code_path": "adapter.py",
            "code_sha256": adapter_digest,
            "builder_symbol": "build_model",
            "dummy_call_symbol": "make_dummy_call",
            "library_recipe": None,
            "code_manifest": code_manifest,
        }
    )
    facts["input_contract"]["args"][0]["shape"] = [1, 2]
    facts["modes"]["meaningful_modes"] = ["eval"]
    facts["external_metadata"]["modes"]["meaningful_modes"] = ["eval"]
    proposal["verified_hashes"]["code"] = adapter_digest
    proposal["verified_hashes"]["code_manifest"] = code_identity
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    artifact = make_proposed_artifact(proposal, {"sources": []}, model_dir)
    environment = real_environment_fixture.binding
    closure = _collect_worker_executable_closure(artifact, environment)
    execution_identity = _execution_identity(
        artifact,
        environment,
        closure_identity=closure.identity,
    )
    manifest = _compile_worker_read_manifest(
        artifact,
        environment,
        execution_identity,
        closure=closure,
    )
    code_members = [{"path": str(adapter), "identity_path": "adapter.py", "sha256": adapter_digest}]
    scratch = tmp_path / "scratch"
    receipt = tmp_path / "result" / "receipt.json"
    request = tmp_path / "request.json"
    request_value = {
        "protocol_version": "menagerie.crawler.worker-request.v3",
        "stable_id": manifest.stable_id,
        "work_id": manifest.work_id,
        "request_nonce": "nonce-v3-supervised",
        "execution_read_manifest_identity": manifest.manifest_id,
        "code_manifest_identity": code_identity,
        "input_identity": "sha256:" + "2" * 64,
        "recipe": {
            "kind": "typed-adapter",
            "path": str(adapter),
            "adapter_sha256": adapter_digest,
            "code_manifest": code_members,
            "code_manifest_sha256": code_identity,
        },
        "modality": "unknown",
        "input_spec": {"shape": [1, 2], "dtype": "float32"},
        "scratch_root": str(scratch),
        "meaningful_modes": ["eval"],
        "mode": "eval",
        "source_identity": "source-v3",
        "execution_identity": execution_identity,
        "recipe_revision": compute_recipe_revision(
            {"recipe_type": "typed-adapter", "path": adapter.name},
            "source-v3",
            adapter_bytes=adapter.read_bytes(),
        ),
        "standard_input_asset": None,
    }
    request.write_text(json.dumps(request_value, sort_keys=True), encoding="utf-8")
    request_identity = hash_bytes(request.read_bytes())
    driver_token = process_start_token(os.getpid())
    assert driver_token is not None
    lease = WorkerLease(
        lease_id="lease-v3-supervised",
        nonce="nonce-v3-supervised",
        run_id="run-v3-supervised",
        stable_id=manifest.stable_id,
        work_id=manifest.work_id,
        request_identity=request_identity,
        execution_identity=execution_identity,
        boot_id=current_boot_id(),
        driver_pid=os.getpid(),
        driver_start_token=driver_token,
        child_pid=None,
        child_start_token=None,
        child_pgid=None,
        receipt_path=receipt,
        opened_at=_timestamp_after(-1),
        deadline_at=_timestamp_after(30),
    )
    lock_path = tmp_path / "locks" / "worker.lock"
    record_path = tmp_path / "locks" / "worker-lease.json"
    lifecycle_events: list[tuple[str, int | None]] = []
    handle = open_worker_lease(
        lock_path,
        record_path,
        lease,
        on_lock_acquired=lambda opened: lifecycle_events.append(("opened", opened.child_pid)),
    )
    try:
        result = supervise_worker(
            request,
            receipt,
            scratch / "supervisor",
            timeout_seconds=20,
            rss_limit_bytes=12 * 1024**3,
            execution_read_manifest=manifest,
            worker_lease_handle=handle,
            on_lease_started=lambda started: lifecycle_events.append(
                ("started", started.child_pid)
            ),
        )
        assert result.receipt_error is None
        assert result.raw_award_receipt_sha256 == stable_hash(result.raw_award_receipt)
        assert result.raw_award_receipt is not None
        assert result.raw_award_receipt["request_sha256"] == request_identity
        assert result.parent_attestation is not None
        assert result.parent_attestation["attestation_version"] == (
            "menagerie.crawler.parent-attestation.v2"
        )
        assert result.parent_attestation["named_raw_award_receipt_sha256"] == (
            result.raw_award_receipt_sha256
        )
        assert handle.lease.child_pid is not None
        assert lifecycle_events == [
            ("opened", None),
            ("started", handle.lease.child_pid),
        ]
        recovered = reconcile_worker_lease(lock_path, record_path)
        assert recovered.state == "completed-before-recovery"
    finally:
        clear_worker_lease(handle)


def test_shutdown_event_is_polled_and_drains_an_honest_observation(tmp_path: Path) -> None:
    """A signal-only event kills the verified group and remains parent-observed."""

    shutdown_event = threading.Event()
    shutdown_event.set()
    observation = run_isolated_subprocess(
        (sys.executable, "-c", "import time; time.sleep(30)"),
        tmp_path / "shutdown",
        timeout_seconds=20,
        shutdown_event=shutdown_event,
    )
    assert observation.shutdown_requested is True
    assert observation.signal_number == 9
    assert observation.timed_out is False


def test_startup_recovery_classifies_a_free_never_started_lease(tmp_path: Path) -> None:
    """Free kernel authority closes pre-spawn metadata without guessing a PID."""

    driver_token = process_start_token(os.getpid())
    assert driver_token is not None
    lease = WorkerLease(
        lease_id="lease-never-started",
        nonce="nonce-never-started",
        run_id="run-never-started",
        stable_id="m_never_started",
        work_id="work-never-started",
        request_identity="sha256:" + "1" * 64,
        execution_identity="sha256:" + "2" * 64,
        boot_id=current_boot_id(),
        driver_pid=os.getpid(),
        driver_start_token=driver_token,
        child_pid=None,
        child_start_token=None,
        child_pgid=None,
        receipt_path=tmp_path / "receipt.json",
        opened_at=_timestamp_after(-1),
        deadline_at=_timestamp_after(30),
    )
    lock_path = tmp_path / "locks" / "worker.lock"
    record_path = tmp_path / "locks" / "worker-lease.json"
    handle = open_worker_lease(lock_path, record_path, lease)
    assert handle.lock_fd is not None
    with pytest.raises(RuntimeError, match="already held"):
        open_worker_lease(lock_path, tmp_path / "locks" / "second.json", lease)
    os.close(handle.lock_fd)
    handle.lock_fd = None
    try:
        recovery = reconcile_worker_lease(lock_path, record_path)
        assert recovery.state == "never-started"
        assert recovery.lock_held is False
        assert recovery.reaped is False
    finally:
        clear_worker_lease(handle)


def test_sigkill_driver_restart_never_double_launches_detached_lock_holder(
    tmp_path: Path,
) -> None:
    """A killed driver cannot outrank the detached child's inherited flock.

    Parameters
    ----------
    tmp_path:
        Isolated process, lock, and durable lease root.
    """

    lock_path = tmp_path / "locks" / "worker.lock"
    record_path = tmp_path / "locks" / "worker-lease.json"
    marker_path = tmp_path / "worker-starts.txt"
    driver_script = r"""
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from menagerie.crawler.authority import WorkerLease
import menagerie.crawler.worker_supervisor as supervisor

lock_path = Path(sys.argv[1])
record_path = Path(sys.argv[2])
marker_path = Path(sys.argv[3])
now = datetime.now(timezone.utc)
token = supervisor.process_start_token(os.getpid())
assert token is not None
lease = WorkerLease(
    lease_id="lease-killed-driver",
    nonce="nonce-killed-driver",
    run_id="run-killed-driver",
    stable_id="m_killed_driver",
    work_id="work-killed-driver",
    request_identity="sha256:" + "1" * 64,
    execution_identity="sha256:" + "2" * 64,
    boot_id=supervisor.current_boot_id(),
    driver_pid=os.getpid(),
    driver_start_token=token,
    child_pid=None,
    child_start_token=None,
    child_pgid=None,
    receipt_path=record_path.with_name("receipt.json"),
    opened_at=(now - timedelta(seconds=1)).isoformat().replace("+00:00", "Z"),
    deadline_at=(now + timedelta(seconds=30)).isoformat().replace("+00:00", "Z"),
)
handle = supervisor.open_worker_lease(lock_path, record_path, lease)
assert handle.lock_fd is not None
child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys,time; "
        "Path(sys.argv[1]).open('a', encoding='utf-8').write('worker\\n'); time.sleep(60)",
        str(marker_path),
    ],
    pass_fds=(handle.lock_fd,),
    start_new_session=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
child_token = supervisor.process_start_token(child.pid)
assert child_token is not None
started = replace(
    lease,
    child_pid=child.pid,
    child_start_token=child_token,
    child_pgid=child.pid,
)
supervisor._atomic_write_worker_lease(record_path, started)
print(json.dumps({"child_pid": child.pid}), flush=True)
time.sleep(60)
"""
    driver = subprocess.Popen(
        [sys.executable, "-c", driver_script, str(lock_path), str(record_path), str(marker_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_pid: int | None = None
    try:
        assert driver.stdout is not None
        started = json.loads(driver.stdout.readline())
        child_pid = int(started["child_pid"])
        marker_deadline = time.monotonic() + 2
        while not marker_path.exists() and time.monotonic() < marker_deadline:
            time.sleep(0.01)
        assert marker_path.read_text(encoding="utf-8").splitlines() == ["worker"]

        os.kill(driver.pid, signal.SIGKILL)
        assert driver.wait(timeout=5) == -signal.SIGKILL

        replacement_script = r"""
import json
import sys
from pathlib import Path
from menagerie.crawler.worker_supervisor import reconcile_worker_lease
recovery = reconcile_worker_lease(Path(sys.argv[1]), Path(sys.argv[2]), timeout_seconds=0.05)
if recovery.state not in {"active", "failed-closed"}:
    with Path(sys.argv[3]).open("a", encoding="utf-8") as handle:
        handle.write("replacement\n")
print(json.dumps({"state": recovery.state, "detail": recovery.detail}))
"""
        replacement = subprocess.run(
            [
                sys.executable,
                "-c",
                replacement_script,
                str(lock_path),
                str(record_path),
                str(marker_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        recovery = json.loads(replacement.stdout)
        assert recovery == {
            "state": "active",
            "detail": "verified-child-still-within-lease-deadline",
        }
        assert marker_path.read_text(encoding="utf-8").splitlines() == ["worker"]
    finally:
        if driver.poll() is None:
            driver.kill()
            driver.wait(timeout=5)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    ("attack", "expected_detail"),
    (
        ("pid-reuse", "child-start-token-mismatch"),
        ("pgid-mismatch", "child-process-group-mismatch"),
        ("stale-boot", "held-lock-from-stale-boot"),
        ("missing-child-identity", "held-lock-without-child-identity"),
        ("corrupt-identity", "corrupt-worker-lease:JSONDecodeError"),
        ("unreadable-process-facts", "child-start-token-mismatch"),
    ),
)
def test_held_worker_lock_identity_mismatches_fail_closed_without_guessed_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
    expected_detail: str,
) -> None:
    """Unverifiable A-02 recovery facts block admission and never guess a kill.

    Parameters
    ----------
    tmp_path:
        Isolated held-lock root.
    monkeypatch:
        Pytest patch helper used to observe forbidden guessed signals.
    attack, expected_detail:
        One corrupted identity relationship and its exact bounded-recovery result.
    """

    with _detached_lock_holder(tmp_path) as (_child, started, lock_path, record_path):
        assert started.child_pgid is not None
        if attack == "pid-reuse":
            worker_supervisor_module._atomic_write_worker_lease(
                record_path, replace(started, child_start_token="reused-pid-token")
            )
        elif attack == "pgid-mismatch":
            worker_supervisor_module._atomic_write_worker_lease(
                record_path, replace(started, child_pgid=started.child_pgid + 1)
            )
        elif attack == "stale-boot":
            worker_supervisor_module._atomic_write_worker_lease(
                record_path, replace(started, boot_id="stale-boot-id")
            )
        elif attack == "missing-child-identity":
            worker_supervisor_module._atomic_write_worker_lease(
                record_path,
                replace(started, child_start_token=None, child_pgid=None),
            )
        elif attack == "corrupt-identity":
            record_path.write_text("{not-json\n", encoding="utf-8")
        elif attack == "unreadable-process-facts":
            monkeypatch.setattr(worker_supervisor_module, "process_start_token", lambda _pid: None)
        else:
            raise AssertionError(f"unsupported identity attack: {attack}")

        killpg_calls: list[tuple[int, int]] = []
        monkeypatch.setattr(
            worker_supervisor_module.os,
            "killpg",
            lambda pgid, signal_number: killpg_calls.append((pgid, signal_number)),
        )
        recovery = reconcile_worker_lease(
            lock_path,
            record_path,
            timeout_seconds=0.05,
            poll_seconds=0.01,
        )
        assert recovery.state == "failed-closed"
        assert recovery.detail == expected_detail
        assert recovery.lock_held is True
        assert recovery.reaped is False
        assert killpg_calls == []
        with pytest.raises(RuntimeError, match="worker lock is already held"):
            open_worker_lease(
                lock_path,
                tmp_path / "locks" / "replacement-lease.json",
                _test_lease(tmp_path, lease_id=f"replacement-{attack}"),
            )


def _adapter_revision(path: Path, source_identity: str = "source-test") -> str:
    """Return the revision for the exact current adapter bytes.

    Parameters
    ----------
    path:
        Adapter source path.
    source_identity:
        Source identity bound into the recipe.

    Returns
    -------
    str
        Exact typed-adapter recipe revision.
    """

    return compute_recipe_revision(
        {"recipe_type": "typed-adapter", "path": path.name},
        source_identity,
        adapter_bytes=path.read_bytes(),
    )


def _tiny_adapter(constructor_body: str = "return Tiny()") -> str:
    """Return a complete tiny typed adapter with a configurable constructor body.

    Parameters
    ----------
    constructor_body:
        Indented-body text placed in ``build_model``.

    Returns
    -------
    str
        Complete adapter source.
    """

    body = "\n".join(f"    {line}" for line in constructor_body.splitlines())
    return f"""from __future__ import annotations
import torch

class Tiny(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1

def build_model() -> object:
{body}

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 2, device=device),), {{}})
"""


def _worker_request(
    adapter: Path,
    tmp_path: Path,
    expected_revision: str,
    *,
    expected_adapter_sha256: str | None = None,
) -> WorkerRequest:
    """Build one eval-only typed worker request.

    Parameters
    ----------
    adapter:
        Typed adapter source path.
    tmp_path:
        Per-test writable root.
    expected_revision:
        Parent-authorized adapter revision.
    expected_adapter_sha256:
        Parent-authorized adapter digest, defaulting to the current bytes.

    Returns
    -------
    WorkerRequest
        Closed worker request.
    """

    return WorkerRequest(
        stable_id="m_round3",
        recipe={
            "kind": "typed-adapter",
            "path": str(adapter),
            "adapter_sha256": expected_adapter_sha256 or hash_bytes(adapter.read_bytes()),
        },
        modality="unknown",
        input_spec=InputSpec((1, 2), "float32"),
        scratch_root=tmp_path / "scratch",
        receipt_path=tmp_path / "result" / "receipt.json",
        meaningful_modes=(RunMode.EVAL,),
        source_identity="source-test",
        recipe_revision=expected_revision,
    )


def test_worker_refuses_changed_adapter_and_reports_matching_observation(tmp_path: Path) -> None:
    """Only byte-identical expected adapter source reaches constructor execution."""

    adapter = tmp_path / "adapter.py"
    adapter.write_text(_tiny_adapter(), encoding="utf-8")
    expected_revision = _adapter_revision(adapter)
    expected_digest = hash_bytes(adapter.read_bytes())
    adapter.write_text(_tiny_adapter() + "\n# changed after validation\n", encoding="utf-8")

    mismatch = run_worker(
        _worker_request(
            adapter,
            tmp_path,
            expected_revision,
            expected_adapter_sha256=expected_digest,
        )
    )

    assert mismatch["constructor_started"] is False
    assert mismatch["per_mode"] == {}
    assert mismatch["error"]["exception_type"] == "menagerie.crawler.recipe.RecipeError"
    assert "digest mismatch" in mismatch["error"]["message"]

    matching_root = tmp_path / "matching"
    matching_root.mkdir()
    matching_adapter = matching_root / "adapter.py"
    matching_adapter.write_text(_tiny_adapter(), encoding="utf-8")
    matching_revision = _adapter_revision(matching_adapter)
    receipt = run_worker(_worker_request(matching_adapter, matching_root, matching_revision))

    assert receipt["error"] is None
    assert receipt["per_mode"]["eval"]["forward_completed"] is True
    assert receipt["recipe_revision"] == matching_revision
    assert receipt["observed_recipe_revision"] == matching_revision
    assert receipt["observed_adapter_sha256"] == hash_bytes(matching_adapter.read_bytes())


def test_worker_refuses_changed_recursive_helper_before_import(tmp_path: Path) -> None:
    """Every request-bound helper byte is rehashed before adapter import executes."""

    helper = tmp_path / "helper.py"
    helper.write_text("INCREMENT = 1\n", encoding="utf-8")
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        _tiny_adapter()
        .replace(
            "import torch\n",
            "import torch\nfrom helper import INCREMENT\n",
        )
        .replace("return value + 1", "return value + INCREMENT"),
        encoding="utf-8",
    )
    members = [
        {"path": "adapter.py", "sha256": hash_bytes(adapter.read_bytes())},
        {"path": "helper.py", "sha256": hash_bytes(helper.read_bytes())},
    ]
    request = _worker_request(adapter, tmp_path, _adapter_revision(adapter))
    assert isinstance(request.recipe, dict)
    request.recipe.update(
        {
            "code_manifest": [
                {
                    "path": str(tmp_path / member["path"]),
                    "identity_path": member["path"],
                    "sha256": member["sha256"],
                }
                for member in members
            ],
            "code_manifest_sha256": stable_hash(members),
        }
    )
    helper.write_text("INCREMENT = 2\n", encoding="utf-8")

    receipt = run_worker(request)

    assert receipt["constructor_started"] is False
    assert receipt["observed_code_manifest_sha256"] != stable_hash(members)
    assert receipt["error"]["exception_type"] == "menagerie.crawler.recipe.RecipeError"
    assert "helper.py" in receipt["error"]["message"]


@pytest.mark.parametrize("suffix", [".bin", ".npz", ".pkl"])
def test_python_undeclared_weight_reads_poison_receipt(tmp_path: Path, suffix: str) -> None:
    """Every undeclared model-data read is denied independently of its suffix.

    Parameters
    ----------
    suffix:
        Representative hidden-weight container suffix.
    """

    hidden = tmp_path / f"weights{suffix}"
    hidden.write_bytes(b"not authorized model data")
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        _tiny_adapter(f"Path({str(hidden)!r}).read_bytes()\nreturn Tiny()").replace(
            "import torch\n", "import torch\nfrom pathlib import Path\n"
        ),
        encoding="utf-8",
    )

    receipt = run_worker(_worker_request(adapter, tmp_path, _adapter_revision(adapter)))

    assert receipt["constructor_completed"] is False
    assert receipt["policy_observation"]["checkpoint_or_weight_read_attempted"] is True
    assert str(hidden) in receipt["policy_observation"]["checkpoint_paths"]
    assert receipt["error"]["reason_code"] == "checkpoint-read"


def test_pretrained_disable_fields_require_real_disabled_constructor_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real disabled parameter passes while absent or enabled claims are refused."""

    module = ModuleType("round3_recipe_fixture")

    def constructor(*, width: int, weights: object | None = None) -> dict[str, object]:
        """Return received constructor values for the declarative loader fixture."""

        return {"width": width, "weights": weights}

    setattr(module, "ExampleNet", constructor)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    base = {
        "distribution": "round3-fixture",
        "version": "1",
        "module": module.__name__,
        "symbol": "ExampleNet",
        "kwargs": {"width": 4, "weights": None},
        "pretrained_disable_fields": ["weights"],
    }

    loaded = load_declarative_recipe(base)

    assert loaded.build_model() == {"width": 4, "weights": None}
    with pytest.raises(RecipeError, match="absent from constructor kwargs"):
        load_declarative_recipe({**base, "pretrained_disable_fields": ["pretrained"]})
    with pytest.raises(RecipeError, match="does not carry a disabling value"):
        load_declarative_recipe({**base, "kwargs": {"width": 4, "weights": True}})


def test_proposal_refuses_bogus_pretrained_disable_field(tmp_path: Path) -> None:
    """R1 proposal validation binds every disable claim to an explicit disabled kwarg."""

    proposal, manifest = _ground_proposal(tmp_path)
    recipe = proposal["proposed_facts"]["implementation"]["library_recipe"]
    recipe["pretrained_disable_fields"] = ["pretrained"]

    with pytest.raises(ProposalValidationError, match="absent from constructor kwargs"):
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest=manifest,
        )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux syscall-broker regression")
def test_native_undeclared_weight_probe_is_denied_and_reported_as_an_attempt(
    tmp_path: Path,
) -> None:
    """A namespace-denied libc probe returns no bytes and remains separate telemetry."""

    if shutil.which("strace") is None:
        pytest.skip("strace is unavailable")
    hidden = tmp_path / "native-weights.bin"
    hidden.write_bytes(b"native hidden weights")
    adapter = tmp_path / "adapter.py"
    constructor = (
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        f"descriptor = libc.open({str(hidden)!r}.encode(), os.O_RDONLY)\n"
        "if descriptor >= 0:\n"
        "    libc.close(descriptor)\n"
        "return Tiny()"
    )
    adapter.write_text(
        _tiny_adapter(constructor).replace(
            "import torch\n", "import ctypes\nimport os\nimport torch\n"
        ),
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"
    receipt_path = tmp_path / "result" / "receipt.json"
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "stable_id": "m_native_hidden_read",
                "recipe": {
                    "kind": "typed-adapter",
                    "path": str(adapter),
                    "adapter_sha256": hash_bytes(adapter.read_bytes()),
                },
                "modality": "unknown",
                "input_spec": {"shape": [1, 2], "dtype": "float32"},
                "scratch_root": str(scratch),
                "meaningful_modes": ["eval"],
                "source_identity": "source-test",
                "recipe_revision": _adapter_revision(adapter),
            }
        ),
        encoding="utf-8",
    )

    result = supervise_worker(
        request_path,
        receipt_path,
        scratch / "supervisor",
        timeout_seconds=20,
        rss_limit_bytes=12 * 1024**3,
    )

    if result.receipt_error == "failed:sandbox-unavailable":
        pytest.skip("working Linux OS sandbox is unavailable")
    assert result.worker_receipt is None
    assert result.receipt_error == "invalid-receipt:worker-result-envelope"
    assert str(hidden) in result.observation.failed_read_probe_paths
    assert result.success_attestation_sha256 is None


@pytest.mark.parametrize("failure", ["missing", "truncated", "replaced"])
def test_parent_telemetry_integrity_failures_poison_closed(tmp_path: Path, failure: str) -> None:
    """Missing, truncated, or replaced syscall telemetry is always policy poison.

    Parameters
    ----------
    failure:
        Simulated parent telemetry integrity failure.
    """

    audit = tmp_path / "parent-audit.log"
    expected_identity: tuple[int, int] | None = None
    if failure != "missing":
        audit.write_text('1 openat(AT_FDCWD, "/tmp/x", O_RDONLY) = 3\n', encoding="utf-8")
        status = audit.stat()
        expected_identity = (status.st_dev, status.st_ino)
        audit.with_name(f"{audit.name}.anchor").hardlink_to(audit)
    if failure == "replaced":
        replacement = tmp_path / "replacement.log"
        replacement.write_text("1 +++ exited with 0 +++\n", encoding="utf-8")
        replacement.replace(audit)

    observation = _parse_linux_denial_audit(
        audit,
        tmp_path,
        (tmp_path / "scratch",),
        expected_identity=expected_identity,
    )

    assert observation.poisoned is True
    assert observation.checkpoint_or_weight_read_attempted is True
    assert observation.telemetry_failure == failure


def test_parent_telemetry_path_is_outside_child_writable_roots(tmp_path: Path) -> None:
    """The broker log is a parent-owned sibling, never a writable child bind."""

    scratch = tmp_path / "scratch" / "supervisor"
    result = tmp_path / "scratch" / "result"
    scratch.mkdir(parents=True)
    result.mkdir(parents=True)
    audit, _identity = _parent_owned_audit_path(scratch, (scratch, result))

    assert not audit.is_relative_to(scratch)
    assert not audit.is_relative_to(result)
    assert audit.name == "sandbox-syscalls.log"


def test_macos_caught_native_read_denial_is_policy_poison() -> None:
    """Seatbelt file-read-data denials poison a caught native model-data read."""

    observation = _macos_denial_audit(
        b"sandbox-exec: deny(1) file-read-data /tmp/hidden-native-weights.bin\n"
    )

    assert observation.poisoned is True
    assert observation.checkpoint_or_weight_read_attempted is True
    assert "hidden-native-weights.bin" in observation.checkpoint_paths[0]


def _native_read_adapter(declared_input: Path, hidden_input: Path) -> str:
    """Return an adapter that verifies a declared read and catches a denied hidden read.

    Parameters
    ----------
    declared_input:
        Exact parent-declared input that must remain readable.
    hidden_input:
        Existing host file that must be absent from the child namespace.

    Returns
    -------
    str
        Complete typed-adapter source.
    """

    return f"""from __future__ import annotations
import ctypes
import os
import torch

class Tiny(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1

def _native_bytes(path: str) -> bytes | None:
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.open(path.encode(), os.O_RDONLY)
    if descriptor < 0:
        return None
    try:
        buffer = ctypes.create_string_buffer(128)
        length = libc.read(descriptor, buffer, len(buffer))
        return buffer.raw[:length]
    finally:
        libc.close(descriptor)

def build_model() -> object:
    if _native_bytes({str(declared_input)!r}) != b"declared-input":
        raise RuntimeError("declared input was not readable")
    if _native_bytes({str(hidden_input)!r}) is not None:
        raise RuntimeError("undeclared host bytes escaped the namespace")
    return Tiny()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 2, device=device),), {{}})
"""


@pytest.mark.skipif(sys.platform != "linux", reason="Linux minimal-namespace regression")
def test_linux_author_supplied_absolute_input_grant_is_refused(
    tmp_path: Path,
) -> None:
    """Raw input_contract paths cannot expand the parent-owned read allowlist."""

    declared_input = tmp_path / "declared-input.bin"
    hidden_input = tmp_path / "private-host-weights.bin"
    declared_input.write_bytes(b"declared-input")
    hidden_input.write_bytes(b"private-model-weights")
    adapter = tmp_path / "adapter.py"
    adapter.write_text(_native_read_adapter(declared_input, hidden_input), encoding="utf-8")
    source_identity = "source-round4-b"
    recipe_revision = compute_recipe_revision(
        {"recipe_type": "typed-adapter", "path": adapter.name},
        source_identity,
        adapter_bytes=adapter.read_bytes(),
    )
    scratch = tmp_path / "scratch"
    receipt_path = tmp_path / "result" / "receipt.json"
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "stable_id": "m_round4_default_deny",
                "recipe": {
                    "kind": "typed-adapter",
                    "path": str(adapter),
                    "adapter_sha256": hash_bytes(adapter.read_bytes()),
                },
                "input_contract": {"code_path": str(declared_input)},
                "modality": "unknown",
                "input_spec": {"shape": [1, 2], "dtype": "float32"},
                "scratch_root": str(scratch),
                "meaningful_modes": ["eval"],
                "source_identity": source_identity,
                "recipe_revision": recipe_revision,
            }
        ),
        encoding="utf-8",
    )

    result = supervise_worker(
        request_path,
        receipt_path,
        scratch / "supervisor",
        timeout_seconds=20,
        rss_limit_bytes=12 * 1024**3,
    )

    if detect_os_sandbox("Linux") is None or shutil.which("strace") is None:
        assert result.worker_receipt is None
        assert result.receipt_error == "failed:sandbox-unavailable"
        return
    assert result.observation.exit_code == 1
    assert result.worker_receipt is None
    assert result.receipt_error == "invalid-receipt:worker-result-envelope"
    assert str(declared_input) in result.observation.failed_read_probe_paths
    assert result.success_attestation_sha256 is None


def _successful_receipt(path: Path) -> None:
    """Write a self-hashed successful receipt fixture for supervisor poisoning.

    Parameters
    ----------
    path:
        Receipt path to create.
    """

    policy = {
        "network_attempted": False,
        "socket_targets": [],
        "checkpoint_or_weight_read_attempted": False,
        "checkpoint_paths": [],
        "write_outside_scratch_attempted": False,
        "write_paths": [],
        "credentials_present": False,
        "torchlens_import_attempted": False,
        "cache_read_attempted": False,
    }
    payload = {
        "receipt_version": "menagerie.crawler.worker-receipt.v1",
        "policy_observation": policy,
        "error": None,
        "per_mode": {"eval": {"forward_completed": True, "error": None}},
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(make_worker_result_v3_mapping(payload)), encoding="utf-8")


@pytest.mark.parametrize(
    ("failure", "telemetry", "expected_failure"),
    [
        ("missing", None, "missing"),
        ("empty", b"", "empty"),
        ("truncated", b'{"eventMessage":"sandbox deny file-read-data /private/x"}\n', "truncated"),
        (
            "unparseable",
            f"not-an-audit-record\n{_MACOS_AUDIT_COMPLETION_MARKER}\n".encode("ascii"),
            "unparsable-record",
        ),
    ],
)
def test_macos_invalid_parent_audit_telemetry_poisoned_closed(
    tmp_path: Path,
    failure: str,
    telemetry: bytes | None,
    expected_failure: str,
) -> None:
    """Missing, empty, truncated, or malformed parent telemetry cannot permit a run.

    Parameters
    ----------
    failure:
        Human-readable telemetry fixture name.
    telemetry:
        Bytes to write, or ``None`` to simulate a missing channel.
    expected_failure:
        Expected fail-closed integrity diagnosis.
    """

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    audit_path, identity = _parent_owned_audit_path(
        scratch,
        (scratch,),
        filename=f"macos-{failure}.ndjson",
    )
    if telemetry is None:
        audit_path.unlink()
    else:
        audit_path.write_bytes(telemetry)

    observation = _parse_macos_denial_audit(
        audit_path,
        expected_identity=identity,
    )

    assert observation.poisoned is True
    assert observation.telemetry_failure == expected_failure
    receipt_path = tmp_path / "result" / f"{failure}.json"
    _successful_receipt(receipt_path)
    assert poison_receipt_for_sandbox_denial(receipt_path, observation) is True
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    diagnostic = receipt["diagnostic"]
    assert diagnostic["policy_observation"]["checkpoint_or_weight_read_attempted"] is True
    assert diagnostic["error"]["reason_code"] == "checkpoint-read"


def test_macos_only_completion_marked_clean_parent_channel_is_clean() -> None:
    """A verified completion marker permits a clean audit while a caught denial poisons it."""

    clean = _macos_denial_audit((_MACOS_AUDIT_COMPLETION_MARKER + "\n").encode("ascii"))
    denied = _macos_denial_audit(
        (
            '{"eventMessage":"sandbox deny file-read-data /private/hidden.bin'
            '\\nProcess: worker [42]\\nMetaData: long forced telemetry"}\n'
            f"{_MACOS_AUDIT_COMPLETION_MARKER}\n"
        ).encode("utf-8")
    )

    assert clean.poisoned is False
    assert denied.poisoned is True
    assert denied.checkpoint_or_weight_read_attempted is True
    assert denied.checkpoint_paths == ("/private/hidden.bin",)
