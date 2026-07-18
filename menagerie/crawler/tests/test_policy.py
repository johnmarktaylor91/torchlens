"""Tests for execution-only security policy."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from menagerie.crawler.legacy_manifest_audit import (
    audit_runtime_root_grants,
    compile_execution_read_manifest,
)
from menagerie.crawler.policy import ExecutionPolicy, PolicyViolation, build_safe_environment


def test_safe_environment_scrubs_credentials_and_socket_tripwire_fails(tmp_path: Path) -> None:
    """Child environment omits secrets and any socket attempt is observed and rejected."""

    safe = build_safe_environment(
        tmp_path / "safe",
        base_environment={
            "PATH": "/bin",
            "SECRET_TOKEN": "never",  # pragma: allowlist secret
        },
    )
    assert safe["PATH"] == "/bin"
    assert "SECRET_TOKEN" not in safe
    assert safe["HF_HUB_OFFLINE"] == "1"

    with ExecutionPolicy(tmp_path / "scratch") as observation:
        with pytest.raises(PolicyViolation, match="blocked socket"):
            socket.create_connection(("example.invalid", 443))
        with socket.socket() as client:
            with pytest.raises(PolicyViolation, match="blocked socket"):
                client.connect(("example.invalid", 443))
    assert observation.network_attempted is True
    assert len(observation.socket_targets) == 2


def test_legacy_root_grant_is_audit_only(tmp_path: Path) -> None:
    """The quarantined v1 parser may characterize history but cannot reach live spawn."""

    legacy_root = tmp_path.resolve()
    manifest = compile_execution_read_manifest(
        stable_id="m_legacy_audit",
        work_id="work-legacy-audit",
        execution_identity="sha256:" + "1" * 64,
        code_manifest_identity="sha256:" + "2" * 64,
        code_members=(),
        runtime_support=((legacy_root, "runtime-root"),),
    )
    assert audit_runtime_root_grants(manifest) == (legacy_root,)
