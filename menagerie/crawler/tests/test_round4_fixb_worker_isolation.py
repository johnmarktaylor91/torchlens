"""Regression coverage for round-4 Group B worker isolation fixes."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from menagerie.crawler.identity import compute_recipe_revision, hash_bytes, stable_hash
from menagerie.crawler.policy import detect_os_sandbox
from menagerie.crawler.worker_supervisor import (
    _MACOS_AUDIT_COMPLETION_MARKER,
    _macos_denial_audit,
    _parent_owned_audit_path,
    _parse_macos_denial_audit,
    poison_receipt_for_sandbox_denial,
    supervise_worker,
)


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
def test_linux_undeclared_native_read_is_denied_before_bytes_are_returned(
    tmp_path: Path,
) -> None:
    """The minimal namespace permits declared runtime data but never returns hidden bytes."""

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
    assert result.observation.exit_code == 0
    assert result.worker_receipt is not None
    assert result.worker_receipt["constructor_completed"] is True
    assert result.worker_receipt["per_mode"]["eval"]["forward_completed"] is True
    policy = result.worker_receipt["policy_observation"]
    assert policy["checkpoint_or_weight_read_attempted"] is True
    assert str(hidden_input) in policy["checkpoint_paths"]
    assert result.worker_receipt["error"]["reason_code"] == "checkpoint-read"


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
    path.write_text(
        json.dumps({**payload, "receipt_sha256": stable_hash(payload)}),
        encoding="utf-8",
    )


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
    assert receipt["policy_observation"]["checkpoint_or_weight_read_attempted"] is True
    assert receipt["error"]["reason_code"] == "checkpoint-read"


def test_macos_only_completion_marked_clean_parent_channel_is_clean() -> None:
    """A verified completion marker permits a clean audit while a caught denial poisons it."""

    clean = _macos_denial_audit((_MACOS_AUDIT_COMPLETION_MARKER + "\n").encode("ascii"))
    denied = _macos_denial_audit(
        (
            '{"eventMessage":"sandbox deny file-read-data /private/hidden.bin"}\n'
            f"{_MACOS_AUDIT_COMPLETION_MARKER}\n"
        ).encode("utf-8")
    )

    assert clean.poisoned is False
    assert denied.poisoned is True
    assert denied.checkpoint_or_weight_read_attempted is True
    assert "hidden.bin" in denied.checkpoint_paths[0]
