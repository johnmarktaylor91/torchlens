"""Round-9 execution authority, attestation, and isolation regressions."""

from __future__ import annotations

import json
import shutil
import socket
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

import pytest

from menagerie.crawler.checkpoint import _externally_controlled_record_text
from menagerie.crawler.constants import RunMode
from menagerie.crawler.driver import _redact_attempt_diagnostics, _supervised_failure
from menagerie.crawler.identity import (
    canonical_json_bytes,
    compute_recipe_revision,
    hash_bytes,
    stable_hash,
)
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.policy import ExecutionPolicy, PolicyViolation, detect_os_sandbox
from menagerie.crawler.proposal import (
    ProposalValidationError,
    _validate_author_read_grants,
)
from menagerie.crawler.reducer import (
    CanonicalReducer,
    ReductionError,
    _parent_success_attestation_matches,
    expected_standard_asset,
)
from menagerie.crawler.schema import validate_payload
from menagerie.crawler.standard_inputs import InputSpec
from menagerie.crawler.tests.conftest import (
    make_attempt,
    make_author_proposal,
    make_gate,
    make_model,
)
from menagerie.crawler.worker import WorkerRequest, run_worker
from menagerie.crawler.worker_supervisor import (
    _MACOS_AUDIT_COMPLETION_MARKER,
    _MacOSAuditChannel,
    _finish_macos_denial_audit,
    _macos_denial_audit,
    _parse_linux_denial_audit,
    _request_allowed_read_paths,
    supervise_worker,
)


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
    assert result.worker_receipt is not None
    assert result.receipt_error == "missing-parent-success-attestation"
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
    assert result.worker_receipt is not None
    assert result.receipt_error is None
    receipt = result.worker_receipt
    failure = _supervised_failure(result, receipt, {}, receipt["policy_observation"])
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

    honest_success = make_attempt("m_honest_success")
    assert _parent_success_attestation_matches(honest_success)


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
    with CanonicalReducer(paths, stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        reducer.append_attempt(clean)
        reducer.append_attempt(dirty)
        with pytest.raises(ReductionError, match="clean successful worker receipt"):
            reducer.append_model(model)


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
    model = make_model(accepted=True)
    model["observed"]["input_asset"] = None
    model["observed"]["input_kind"] = "random-fallback"
    with CanonicalReducer(paths, stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        reducer.append_attempt(attempt)
        assert reducer.append_model(model).appended


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
    with pytest.raises(ProposalValidationError, match="repository-relative"):
        _validate_author_read_grants(facts, tmp_path / "model")
    facts["input_contract"]["code_path"] = None
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


def test_macos_audit_finish_drains_delayed_denial_before_completion(tmp_path: Path) -> None:
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
