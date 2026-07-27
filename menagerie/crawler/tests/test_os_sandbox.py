"""Tests for fail-closed OS-boundary worker isolation."""

from __future__ import annotations

import builtins
import socket
import sys
from pathlib import Path

import pytest

from menagerie.crawler import worker_supervisor
from menagerie.crawler.policy import (
    ExecutionPolicy,
    PolicyViolation,
    _linux_host_transport_library_capability,
    _linux_minimal_read_mounts,
    detect_os_sandbox,
    generate_macos_sandbox_profile,
)
from menagerie.crawler.worker_supervisor import run_isolated_subprocess, supervise_worker

import json
import shutil
import threading
import time
from copy import deepcopy
import menagerie.crawler.worker_supervisor as supervisor_module
from menagerie.crawler.checkpoint import _externally_controlled_record_text
from menagerie.crawler.constants import RunMode
from menagerie.crawler.driver import _redact_attempt_diagnostics
from menagerie.crawler.identity import (
    canonical_json_bytes,
    compute_recipe_revision,
    hash_bytes,
    stable_hash,
)
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.proposal import (
    ProposalValidationError,
    _validate_author_read_grants,
)
from menagerie.crawler.reducer import (
    CanonicalReducer,
    ReductionError,
    expected_standard_asset,
)
from menagerie.crawler.schema import validate_payload
from menagerie.crawler.standard_inputs import InputSpec
from menagerie.crawler.tests.conftest import (
    make_attempt,
    make_author_proposal,
    make_authority_context,
    make_gate,
    make_model,
    make_worker_result_v3_mapping,
    rebind_attempt_raw_proof,
)
from menagerie.crawler.worker import WorkerRequest, run_worker
from menagerie.crawler.worker_supervisor import (
    _MACOS_AUDIT_COMPLETION_MARKER,
    _MacOSAuditChannel,
    _finish_macos_denial_audit,
    _macos_denial_audit,
    _parse_linux_denial_audit,
    _request_allowed_read_paths,
    SupervisorObservation,
)


def test_macos_profile_denies_network_and_writes_except_designated_roots(
    tmp_path: Path,
) -> None:
    """The generated Seatbelt profile has deterministic deny-first boundary rules."""

    scratch = tmp_path / "scratch"
    result = tmp_path / "result root"
    profile = generate_macos_sandbox_profile((result, scratch, scratch / "nested"))

    # Every read allowance names the exact file-read-data operation, never file-read*.
    # Seatbelt resolves a request against the most specific matching operation node, so a
    # file-read* allowance is silently inert against the exact (deny file-read-data) above it
    # and the profile would deny every read on the host, aborting the child inside dyld.
    assert profile == (
        "(version 1)\n"
        "(allow default)\n"
        "(deny network*)\n"
        "(deny file-read-data)\n"
        "(deny file-write*)\n"
        '(allow file-read-data (require-all (vnode-type DIRECTORY) (literal "/")))\n'
        '(allow file-read-data (subpath "/System"))\n'
        '(allow file-read-data (subpath "/usr/lib"))\n'
        '(allow file-read-data (subpath "/Library/Apple"))\n'
        '(allow file-read-data (subpath "/private/etc"))\n'
        '(allow file-read-data (subpath "/dev"))\n'
        '(allow file-write* (literal "/dev/null"))\n'
        f'(allow file-read-data (literal "{result.resolve()}"))\n'
        f'(allow file-read-data (subpath "{result.resolve()}"))\n'
        f'(allow file-read-data (literal "{scratch.resolve()}"))\n'
        f'(allow file-read-data (subpath "{scratch.resolve()}"))\n'
        f'(allow file-write* (literal "{result.resolve()}"))\n'
        f'(allow file-write* (subpath "{result.resolve()}"))\n'
        f'(allow file-write* (literal "{scratch.resolve()}"))\n'
        f'(allow file-write* (subpath "{scratch.resolve()}"))\n'
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux sandbox integration test")
def test_linux_os_sandbox_wraps_command_and_denies_network_and_outside_write(
    tmp_path: Path,
) -> None:
    """A Linux boundary makes root read-only and provides no outbound network route."""

    sandbox = detect_os_sandbox("Linux")
    if sandbox is None:
        pytest.skip("no working Linux OS sandbox tool installed")
    assert sandbox is not None
    scratch = tmp_path / "scratch"
    outside = tmp_path / "outside.bin"
    script = (
        "import io,socket,sys; "
        f"outside={str(outside)!r}; "
        "write_denied=False; network_denied=False; "
        "\ntry:\n io.open(outside,'wb').write(b'forbidden')"
        "\nexcept OSError:\n write_denied=True"
        "\nclient=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)"
        "\ntry:\n client.sendto(b'probe',('203.0.113.1',9))"
        "\nexcept OSError:\n network_denied=True"
        "\nfinally:\n client.close()"
        "\nprint(f'write_denied={write_denied} network_denied={network_denied}')"
        "\nsys.exit(71 if write_denied and network_denied else 70)"
    )

    observation = run_isolated_subprocess(
        (sys.executable, "-c", script),
        scratch,
        timeout_seconds=10,
        rss_limit_bytes=1024**3,
    )

    assert Path(observation.argv[0]).name == "strace"
    assert str(Path(sandbox.executable).resolve()) in observation.argv
    assert "--unshare-net" in observation.argv or "--net" in observation.argv
    assert observation.exit_code == 71
    assert "write_denied=True network_denied=True" in observation.stdout_tail
    assert not outside.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux transport capability test")
def test_linux_mount_and_parent_audit_share_exact_transport_capability() -> None:
    """Linux mount and audit accept only one interpreter's exact ELF members."""

    interpreter = Path(sys.executable).resolve()
    capability = _linux_host_transport_library_capability(interpreter)
    assert capability.members
    assert capability.digest.startswith("sha256:")
    mounts = _linux_minimal_read_mounts(
        (str(interpreter),),
        Path.cwd(),
        (),
        host_transport_capability=capability,
    )
    prefix = interpreter.parent.parent
    assert all(path in mounts or path.is_relative_to(prefix) for path in capability.members)
    assert all(
        worker_supervisor._system_transport_library_path_allowed(  # noqa: SLF001
            path,
            capability,
        )
        for path in capability.canonical_members
    )

    unlisted = next(
        resolved
        for candidate in sorted(Path("/usr/lib").rglob("*.so*"), key=lambda path: str(path))
        if (resolved := candidate.resolve()).is_file()
        and resolved not in capability.canonical_members
    )
    assert not worker_supervisor._system_transport_library_path_allowed(  # noqa: SLF001
        unlisted,
        capability,
    )


def test_supervisor_fails_closed_when_no_os_sandbox_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sandbox detection failure refuses the worker instead of launching unsandboxed."""

    def no_sandbox(_system_name: str | None = None) -> None:
        """Report no available OS sandbox for the fail-closed branch."""

        return None

    def forbidden_popen(*args: object, **kwargs: object) -> None:
        """Fail if the supervisor attempts any unsandboxed process launch."""

        del args, kwargs
        raise AssertionError("worker must not launch without an OS sandbox")

    monkeypatch.setattr(worker_supervisor, "detect_os_sandbox", no_sandbox)
    monkeypatch.setattr(worker_supervisor.subprocess, "Popen", forbidden_popen)
    scratch = tmp_path / "scratch"
    request = tmp_path / "request.json"
    request.write_text(
        '{"recipe": {}, "input_spec": {"shape": [1], "dtype": "float32"}}',
        encoding="utf-8",
    )
    result = supervise_worker(
        request,
        tmp_path / "result" / "receipt.json",
        scratch,
    )

    assert result.worker_receipt is None
    assert result.receipt_error == "failed:policy"
    assert result.observation.exit_code is None
    assert result.observation.stderr_tail == "failed:policy\n"


def test_python_tripwires_remain_as_secondary_audit(tmp_path: Path) -> None:
    """Python open and socket tripwires still record attempts inside the OS boundary."""

    original_open = builtins.open
    with ExecutionPolicy(tmp_path / "scratch") as observation:
        assert builtins.open != original_open
        with pytest.raises(PolicyViolation, match="blocked write"):
            builtins.open(tmp_path / "outside.txt", "w")
        with pytest.raises(PolicyViolation, match="blocked socket"):
            socket.create_connection(("example.invalid", 443))

    assert observation.write_outside_scratch_attempted is True
    assert observation.network_attempted is True


def _adapter_revision(path: Path, source_identity: str) -> str:
    """Return the exact legacy typed-adapter revision for a test source file.

    Parameters
    ----------
    path:
        Adapter source file.
    source_identity:
        Source identity echoed by the worker.

    Returns
    -------
    str
        Bound typed-adapter recipe revision.
    """

    return compute_recipe_revision(
        {"recipe_type": "typed-adapter", "path": path.name},
        source_identity,
        adapter_bytes=path.read_bytes(),
    )


def _supervisor_request(adapter: Path, root: Path, source_identity: str) -> Path:
    """Write one legacy-compatible worker request for a typed adapter.

    Parameters
    ----------
    adapter, root, source_identity:
        Adapter file, isolated test root, and exact source identity.

    Returns
    -------
    pathlib.Path
        Immutable request JSON path.
    """

    request = root / "request.json"
    request.write_text(
        json.dumps(
            {
                "stable_id": "m_round9_supervised",
                "recipe": {
                    "kind": "typed-adapter",
                    "path": str(adapter),
                    "adapter_sha256": hash_bytes(adapter.read_bytes()),
                },
                "modality": "unknown",
                "input_spec": {"shape": [1, 2], "dtype": "float32"},
                "scratch_root": str(root / "supervisor"),
                "meaningful_modes": ["eval"],
                "mode": "eval",
                "source_identity": source_identity,
                "recipe_revision": _adapter_revision(adapter, source_identity),
                "execution_identity": "execution-round9",
            }
        ),
        encoding="utf-8",
    )
    return request


@pytest.mark.skipif(sys.platform != "linux", reason="Linux real-supervisor regression")
def test_constructor_cannot_self_certify_forged_success_receipt(tmp_path: Path) -> None:
    """An early constructor exit lacks the parent-owned normal-completion attestation."""

    adapter = tmp_path / "forge_adapter.py"
    adapter.write_text(
        """from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from menagerie.crawler.identity import stable_hash

class NeverRun:
    def forward(self, value: object) -> object:
        raise AssertionError("forward must never run")

def build_model() -> object:
    receipt = Path(sys.argv[sys.argv.index("--receipt") + 1])
    payload = {
        "receipt_version": "menagerie.crawler.worker-receipt.v1",
        "error": None,
        "per_mode": {"eval": {"forward_completed": True, "error": None}},
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({**payload, "receipt_sha256": stable_hash(payload)}))
    os._exit(0)

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed, device
    return ((object(),), {})
""",
        encoding="utf-8",
    )
    request = _supervisor_request(adapter, tmp_path, "source-forgery")
    receipt_path = tmp_path / "result" / "receipt.json"

    result = supervise_worker(
        request,
        receipt_path,
        tmp_path / "supervisor",
        timeout_seconds=20,
        rss_limit_bytes=12 * 1024**3,
    )

    if detect_os_sandbox("Linux") is None or shutil.which("strace") is None:
        pytest.skip("working Linux OS sandbox is unavailable")
    assert result.observation.exit_code == 0
    assert result.worker_receipt is None
    assert result.receipt_error == "invalid-receipt:worker-result-envelope"
    assert result.success_attestation_sha256 is None


@pytest.mark.skipif(sys.platform != "linux", reason="Linux real-supervisor regression")
def test_real_supervisor_retains_and_classifies_constructor_failure(tmp_path: Path) -> None:
    """A real failure keeps exact local diagnostics but exposes only redacted records."""

    adapter = tmp_path / "failing_adapter.py"
    adapter.write_text(
        """from __future__ import annotations
import sys

def build_model() -> object:
    print("round9 model stdout secret")
    print("round9 model stderr secret", file=sys.stderr)
    raise RuntimeError("round9 constructor exploded")

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed, device
    return ((object(),), {})
""",
        encoding="utf-8",
    )
    request = _supervisor_request(adapter, tmp_path, "source-failure")
    result = supervise_worker(
        request,
        tmp_path / "result" / "receipt.json",
        tmp_path / "supervisor",
        timeout_seconds=20,
        rss_limit_bytes=12 * 1024**3,
    )

    if detect_os_sandbox("Linux") is None or shutil.which("strace") is None:
        pytest.skip("working Linux OS sandbox is unavailable")
    assert result.observation.exit_code == 1
    assert result.worker_receipt is None
    assert result.receipt_error == "invalid-receipt:worker-result-envelope"
    receipt = json.loads((tmp_path / "result" / "receipt.json").read_text(encoding="utf-8"))
    worker_error = receipt["error"]
    failure = {
        "stage": "constructor",
        "reason_code": "exception",
        "exception_type": worker_error["exception_type"],
        "message": worker_error["message"],
        "traceback": worker_error["traceback"],
        "no_traceback_reason": None,
        "native_crash": False,
        "details": {"receipt_error": worker_error},
    }
    assert failure["stage"] == "constructor"
    assert failure["reason_code"] == "exception"
    assert "round9 constructor exploded" in failure["message"]
    assert "RuntimeError" in failure["traceback"]

    raw_attempt = make_attempt("m_round9_supervised")
    raw_attempt["attempt_id"] = stable_hash("c07-real-supervisor-attempt")
    raw_attempt["result"] = "failed"
    raw_attempt["stage"] = "constructor"
    raw_attempt["mode"] = None
    raw_attempt["supervisor_observation"].update(
        {
            "exit_code": result.observation.exit_code,
            "signal": result.observation.signal_number,
            "stdout_sha256": result.observation.stdout_sha256,
            "stdout_bytes": result.observation.stdout_bytes,
            "stdout_tail": result.observation.stdout_tail,
            "stdout_completion_line": None,
            "stderr_sha256": result.observation.stderr_sha256,
            "stderr_bytes": result.observation.stderr_bytes,
            "stderr_tail": result.observation.stderr_tail,
        }
    )
    raw_attempt["error"] = {
        **failure,
        "root_cause_fingerprint": stable_hash(failure),
    }
    redacted = _redact_attempt_diagnostics(
        raw_attempt,
        result.observation,
        tmp_path / ".crawl-local" / "diagnostics",
    )
    public_bytes = canonical_json_bytes(redacted)
    validate_payload(redacted)
    for forbidden in (
        b"round9 model stdout secret",
        b"round9 model stderr secret",
        b"round9 constructor exploded",
        b"Traceback",
    ):
        assert forbidden not in public_bytes
    assert redacted["error"]["stage"] == "constructor"
    assert redacted["error"]["reason_code"] == "exception"
    assert redacted["supervisor_observation"]["stdout_tail"]["stream_sha256"] == (
        result.observation.stdout_sha256
    )
    sidecar_path = tmp_path / redacted["supervisor_observation"]["stdout_tail"]["local_path"]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["stdout"]["tail"] == result.observation.stdout_tail
    assert sidecar["stderr"]["tail"] == result.observation.stderr_tail
    assert sidecar["externally_controlled_fields"]["$.error.traceback"] == failure["traceback"]
    assert redacted["error"]["traceback"]["content_sha256"] == hash_bytes(
        canonical_json_bytes(sidecar["externally_controlled_fields"]["$.error.traceback"])
    )

    record_path = Path("menagerie/crawler/records/attempts/c07-regression.jsonl")
    assert _externally_controlled_record_text(record_path, public_bytes + b"\n") == ()
    forged = deepcopy(redacted)
    forged["supervisor_observation"]["stdout_tail"] = "raw model-controlled text"
    assert "$[0].supervisor_observation.stdout_tail" in _externally_controlled_record_text(
        record_path, canonical_json_bytes(forged) + b"\n"
    )


def test_reducer_rejects_dirty_nonreference_accepted_attempt(tmp_path: Path) -> None:
    """Every accepted attempt is policy-checked, not only the per-mode reference."""

    paths = LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )
    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    clean = make_attempt(attempt_id="attempt-clean")
    dirty = make_attempt(attempt_id="attempt-dirty")
    dirty["ledger_seq"] = 2
    dirty["attempt_no"] = 2
    dirty["policy_observation"]["credentials_present"] = True
    model = make_model(accepted=True, attempt_id="attempt-clean")
    model["execution"]["accepted_attempt_ids"] = ["attempt-clean", "attempt-dirty"]
    with CanonicalReducer(paths, make_authority_context(stable_ids)) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        reducer.append_attempt(clean)
        with pytest.raises(ReductionError, match="policy_observation.clean_flags"):
            reducer.append_attempt(dirty)


def test_random_fallback_receipt_can_earn_run_with_known_modality(tmp_path: Path) -> None:
    """A shape-driven random fallback is an allowed bound asset outcome."""

    paths = LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )
    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    attempt = make_attempt()
    attempt["worker_receipt"]["observed_input_asset_sha256"] = None
    attempt["worker_receipt"]["input_asset"] = None
    attempt["worker_receipt"]["input_kind"] = "random-fallback"
    rebind_attempt_raw_proof(attempt)
    model = make_model(accepted=True)
    model["observed"]["input_asset"] = None
    model["observed"]["input_kind"] = "random-fallback"
    with CanonicalReducer(paths, make_authority_context(stable_ids)) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        reducer.append_attempt(attempt)
        assert reducer.append_model(reducer.prepare_model(model)).appended


def test_missing_standard_asset_raises_reduction_error_not_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing bundled bytes fail through the reducer's model-local error type."""

    import menagerie.crawler.reducer as reducer_module

    monkeypatch.setattr(reducer_module, "ASSET_ROOT", tmp_path / "missing-assets")
    with pytest.raises(ReductionError, match="standard input asset is unavailable"):
        expected_standard_asset(["vision"])


def test_author_path_leaves_never_become_parent_read_grants(tmp_path: Path) -> None:
    """Absolute input paths are rejected at proposal validation and supervisor grant derivation."""

    proposal = make_author_proposal("m_round9_paths")
    facts = deepcopy(proposal["proposed_facts"])
    hidden = tmp_path / "host-checkpoints"
    hidden.mkdir()
    facts["input_contract"]["code_path"] = str(hidden)
    with pytest.raises(ProposalValidationError, match="forbids code_path presence"):
        _validate_author_read_grants(facts, tmp_path / "model")
    del facts["input_contract"]["code_path"]
    facts["input_contract"]["non_tensor_values"] = [
        {"type": "path", "value": "missing-checkpoint.pt"}
    ]
    with pytest.raises(ProposalValidationError, match="model-local regular file"):
        _validate_author_read_grants(facts, tmp_path / "model")

    adapter = tmp_path / "adapter.py"
    adapter.write_text("# source", encoding="utf-8")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "recipe": {"kind": "typed-adapter", "path": str(adapter)},
                "modality": "unknown",
                "input_contract": {"code_path": str(hidden)},
            }
        ),
        encoding="utf-8",
    )
    allowed = _request_allowed_read_paths((sys.executable, "--request", str(request)))
    assert hidden.resolve() not in allowed
    assert all(path.is_file() for path in allowed)


def test_linux_parser_covers_datagram_and_nonopen_read_attempts(tmp_path: Path) -> None:
    """sendmmsg, readlinkat, mmap-fd, and handle reads all poison policy."""

    hidden = "/tmp/round9-hidden-weights.bin"
    audit = tmp_path / "strace.log"
    audit.write_text(
        "101 sendmmsg(3, [{msg_hdr={msg_name={sa_family=AF_INET, "
        'sin_port=htons(9), sin_addr=inet_addr("203.0.113.1")}}}], 1, 0) '
        "= -1 ENETUNREACH (Network is unreachable)\n"
        f'101 readlinkat(AT_FDCWD, "{hidden}", 0x1, 4096) = -1 EACCES (Permission denied)\n'
        f"101 mmap(NULL, 4096, PROT_READ, MAP_PRIVATE, 7<{hidden}>, 0) = 0x1234\n"
        f"101 pread64(7<{hidden}>, 0x1, 16, 0) = 16\n"
        "101 open_by_handle_at(5, {handle_bytes=8}, O_RDONLY) = -1 EPERM (Operation not permitted)\n"
        "101 +++ exited with 0 +++\n",
        encoding="utf-8",
    )

    observation = _parse_linux_denial_audit(audit, tmp_path, (tmp_path / "scratch",))

    assert observation.network_attempted is True
    assert hidden in observation.checkpoint_paths
    assert "<open_by_handle_at:undeclared-file-handle>" in observation.checkpoint_paths


def test_in_child_policy_blocks_internet_datagram_send(tmp_path: Path) -> None:
    """Python UDP sendto is an attempt even without a preceding connect call."""

    policy = ExecutionPolicy(tmp_path / "scratch")
    datagram = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(PolicyViolation, match="blocked socket sendto"):
            with policy:
                datagram.sendto(b"telemetry", ("203.0.113.1", 9))
    finally:
        datagram.close()
    assert policy.observation.network_attempted is True


def test_macos_denial_parser_scopes_pid_and_ignores_other_denial_classes() -> None:
    """Machine-wide and mach-lookup denial chatter cannot poison one worker."""

    telemetry = (
        json.dumps(
            {
                "processID": 999,
                "processImagePath": "/Applications/Other.app/Contents/MacOS/other",
                "eventMessage": "Sandbox: other(999) deny file-read-data /tmp/other.bin",
            }
        )
        + "\n"
        + json.dumps(
            {
                "processID": 42,
                "eventMessage": "Sandbox: worker(42) deny mach-lookup com.apple.noise",
            }
        )
        + "\n"
        + _MACOS_AUDIT_COMPLETION_MARKER
        + "\n"
    ).encode("utf-8")

    clean = _macos_denial_audit(telemetry, expected_process_ids=(42,))

    assert clean.poisoned is False
    denied = _macos_denial_audit(
        (
            json.dumps(
                {
                    "processID": 43,
                    "parentProcessID": 42,
                    "eventMessage": "Sandbox: child(43) deny mach-lookup com.apple.noise",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "processID": 44,
                    "parentProcessID": 43,
                    "eventMessage": ("Sandbox: grandchild(44) deny network-outbound 203.0.113.1"),
                }
            )
            + "\n"
            + _MACOS_AUDIT_COMPLETION_MARKER
            + "\n"
        ).encode("utf-8"),
        expected_process_ids=(42,),
    )
    assert denied.network_attempted is True
    assert denied.poisoned is True


def test_macos_unattributable_policy_denial_fails_closed() -> None:
    """A policy-class denial without a trustworthy worker/noise scope poisons telemetry."""

    telemetry = (
        json.dumps(
            {
                "processID": 999,
                "eventMessage": "Sandbox: helper(999) deny file-read-data /tmp/hidden.bin",
            }
        )
        + "\n"
        + _MACOS_AUDIT_COMPLETION_MARKER
        + "\n"
    ).encode("utf-8")

    observed = _macos_denial_audit(telemetry, expected_process_ids=(42,))

    assert observed.poisoned is True
    assert observed.telemetry_failure == "unattributable-denial"


def test_macos_descendant_denial_scopes_by_parent_owned_runtime_root(tmp_path: Path) -> None:
    """A first descendant-only denial is scoped without admitting unrelated host noise."""

    runtime_root = tmp_path / "environment"
    runtime_root.mkdir()
    telemetry = (
        json.dumps(
            {
                "processID": 999,
                "processImagePath": "/Applications/Other.app/Contents/MacOS/other",
                "eventMessage": "Sandbox: other(999) deny network-outbound 203.0.113.2",
            }
        )
        + "\n"
        + json.dumps(
            {
                "processID": 44,
                "processImagePath": str(runtime_root / "bin" / "python"),
                "eventMessage": "Sandbox: child(44) deny network-outbound 203.0.113.1",
            }
        )
        + "\n"
        + _MACOS_AUDIT_COMPLETION_MARKER
        + "\n"
    ).encode("utf-8")

    observation = _macos_denial_audit(
        telemetry,
        expected_process_ids=(42,),
        expected_process_roots=(runtime_root,),
    )

    assert observation.network_attempted is True
    assert len(observation.socket_targets) == 1
    assert "203.0.113.1" in observation.socket_targets[0]


def test_macos_audit_finish_drains_delayed_denial_before_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deny delivered just after worker exit is drained before the marker is written."""

    class _CollectorProcess:
        """Minimal completed log-stream process fixture."""

        def terminate(self) -> None:
            """Accept the supervisor's bounded collector shutdown."""

        def wait(self, timeout: float | None = None) -> int:
            """Return the normal SIGTERM status after the drain.

            Parameters
            ----------
            timeout:
                Parent shutdown bound.

            Returns
            -------
            int
                Conventional negative SIGTERM return code.
            """

            del timeout
            return -15

    path = tmp_path / "macos-seatbelt.ndjson"
    handle = path.open("wb")
    status = path.stat()
    channel = _MacOSAuditChannel(
        path,
        (status.st_dev, status.st_ino),
        _CollectorProcess(),  # type: ignore[arg-type]
        handle,
        worker_pid=42,
    )

    def observed_post_exit_sentinel(_channel: _MacOSAuditChannel, phase: str) -> bool:
        """Model a parent sentinel delivered after the delayed worker denial."""

        assert phase == "post-exit"
        time.sleep(0.25)
        return True

    monkeypatch.setattr(
        supervisor_module,
        "_emit_macos_audit_sentinel",
        observed_post_exit_sentinel,
    )

    def delayed_denial() -> None:
        """Append one unified-log record after the worker's observed exit."""

        time.sleep(0.15)
        with path.open("ab") as writer:
            writer.write(
                (
                    json.dumps(
                        {
                            "processID": 42,
                            "eventMessage": (
                                "Sandbox: worker(42) deny file-read-data /tmp/delayed.bin"
                            ),
                        }
                    )
                    + "\n"
                ).encode("utf-8")
            )
            writer.flush()

    writer = threading.Thread(target=delayed_denial)
    writer.start()
    _finish_macos_denial_audit(channel)
    writer.join(timeout=2)

    observation = _macos_denial_audit(path.read_bytes(), expected_process_ids=(42,))
    assert observation.checkpoint_or_weight_read_attempted is True
    assert any("delayed.bin" in value for value in observation.checkpoint_paths)


def test_macos_missing_post_exit_sentinel_is_telemetry_poison(tmp_path: Path) -> None:
    """A collector without a provable post-exit delivery boundary cannot read clean."""

    class _CollectorProcess:
        """Minimal completed log-stream process fixture."""

        def terminate(self) -> None:
            """Accept collector shutdown."""

        def wait(self, timeout: float | None = None) -> int:
            """Return the conventional SIGTERM status."""

            del timeout
            return -15

    path = tmp_path / "macos-seatbelt.ndjson"
    handle = path.open("wb")
    status = path.stat()
    channel = _MacOSAuditChannel(
        path,
        (status.st_dev, status.st_ino),
        _CollectorProcess(),  # type: ignore[arg-type]
        handle,
        worker_pid=42,
    )

    _finish_macos_denial_audit(channel)
    observation = _macos_denial_audit(path.read_bytes(), expected_process_ids=(42,))

    assert observation.poisoned is True
    assert observation.telemetry_failure == "empty"


def test_supervisor_poisons_caught_dirty_policy_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caught child tripwire cannot retain a successful atomic receipt or attestation."""

    receipt_path = tmp_path / "result" / "receipt.json"
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")

    def dirty_success(
        _argv: object,
        scratch_root: Path,
        **_kwargs: object,
    ) -> SupervisorObservation:
        """Write a caught-network receipt while reporting a normal child exit."""

        policy = {
            "network_attempted": True,
            "socket_targets": ["caught.example:443"],
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
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        wrapper = make_worker_result_v3_mapping(payload)
        receipt_path.write_text(json.dumps(wrapper), encoding="utf-8")
        scratch_root.mkdir(parents=True, exist_ok=True)
        stdout_path = scratch_root / "stdout.log"
        stderr_path = scratch_root / "stderr.log"
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        return SupervisorObservation(
            argv=(sys.executable,),
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
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            success_attestation_sha256="a" * 64,
            attested_receipt_sha256=str(wrapper["result_sha256"]),
        )

    monkeypatch.setattr(supervisor_module, "run_isolated_subprocess", dirty_success)
    result = supervise_worker(request_path, receipt_path, tmp_path / "scratch")

    assert result.worker_receipt is not None
    diagnostic = result.worker_receipt["diagnostic"]
    assert diagnostic["error"]["reason_code"] == "network-attempt"
    assert diagnostic["per_mode"]["eval"]["error"]["reason_code"] == "network-attempt"
    assert result.receipt_error == "missing-parent-success-attestation"
    assert result.success_attestation_sha256 is None


def test_declarative_revision_fallback_binds_executed_payload(tmp_path: Path) -> None:
    """Parent identity bytes cannot substitute for a different declarative recipe."""

    accepted_recipe = {
        "distribution": "torch",
        "version": "test",
        "artifact_sha256": "sha256:" + "a" * 64,
        "module": "torch.nn",
        "symbol": "Linear",
        "kwargs": {"in_features": 2, "out_features": 2},
        "pretrained_disable_fields": [],
    }
    executed_recipe = {**accepted_recipe, "symbol": "ReLU", "kwargs": {}}
    identity_payload = {
        "implementation": {"library_recipe": accepted_recipe},
        "input_contract": {},
        "modes": {"meaningful_modes": ["eval"]},
    }
    source_identity = "source-declarative-round9"
    revision = compute_recipe_revision(identity_payload, source_identity)
    request = WorkerRequest(
        stable_id="m_declarative_binding",
        recipe={"kind": "declarative-library", "recipe": executed_recipe},
        modality=None,
        input_spec=InputSpec((1, 2), "float32"),
        scratch_root=tmp_path / "scratch",
        receipt_path=tmp_path / "result" / "receipt.json",
        meaningful_modes=(RunMode.EVAL,),
        source_identity=source_identity,
        recipe_revision=revision,
        recipe_identity_payload=identity_payload,
    )

    receipt = run_worker(request)

    assert receipt["constructor_started"] is False
    assert "do not match the accepted identity payload" in receipt["error"]["message"]
