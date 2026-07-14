"""Tests for execution-only security policy."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

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
