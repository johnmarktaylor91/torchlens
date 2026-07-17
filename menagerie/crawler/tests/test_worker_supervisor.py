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
from menagerie.crawler.identity import compute_recipe_revision, hash_bytes, stable_hash
from menagerie.crawler.policy import compile_execution_read_manifest
from menagerie.crawler.worker_supervisor import (
    clear_worker_lease,
    current_boot_id,
    open_worker_lease,
    process_start_token,
    reconcile_worker_lease,
    run_isolated_subprocess,
    supervise_worker,
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


def test_supervisor_accepts_only_atomic_worker_receipt(tmp_path: Path) -> None:
    """The standard worker succeeds in a fresh process and its receipt hash verifies."""

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
    assert result.receipt_error is None
    assert result.worker_receipt is not None
    assert result.worker_receipt["awards_runs"] is False


def test_v3_worker_binds_raw_receipt_attestation_manifest_and_child_lease(
    tmp_path: Path,
) -> None:
    """One v3 success requires all four frozen security associations."""

    adapter = tmp_path / "adapter.py"
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
    code_members = [{"path": str(adapter), "identity_path": "adapter.py", "sha256": adapter_digest}]
    code_identity = stable_hash([{"path": "adapter.py", "sha256": adapter_digest}])
    execution_identity = "sha256:" + "1" * 64
    manifest = compile_execution_read_manifest(
        stable_id="m_v3_supervised",
        work_id="work-v3-supervised",
        execution_identity=execution_identity,
        code_manifest_identity=code_identity,
        code_members=((adapter, adapter_digest, "python-source"),),
        runtime_support=((Path(__file__).resolve().parents[2], "runtime-root"),),
    )
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
