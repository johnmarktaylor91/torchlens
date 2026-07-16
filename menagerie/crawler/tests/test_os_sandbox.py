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
    detect_os_sandbox,
    generate_macos_sandbox_profile,
)
from menagerie.crawler.worker_supervisor import run_isolated_subprocess, supervise_worker


def test_macos_profile_denies_network_and_writes_except_designated_roots(
    tmp_path: Path,
) -> None:
    """The generated Seatbelt profile has deterministic deny-first boundary rules."""

    scratch = tmp_path / "scratch"
    result = tmp_path / "result root"
    profile = generate_macos_sandbox_profile((result, scratch, scratch / "nested"))

    assert profile == (
        "(version 1)\n"
        "(allow default)\n"
        "(deny network*)\n"
        "(deny file-read-data)\n"
        "(deny file-write*)\n"
        '(allow file-read* (subpath "/System"))\n'
        '(allow file-read* (subpath "/usr/lib"))\n'
        '(allow file-read* (subpath "/Library/Apple"))\n'
        '(allow file-read* (subpath "/private/etc"))\n'
        '(allow file-read* (subpath "/dev"))\n'
        '(allow file-write* (literal "/dev/null"))\n'
        f'(allow file-read* (literal "{result.resolve()}"))\n'
        f'(allow file-read* (subpath "{result.resolve()}"))\n'
        f'(allow file-read* (literal "{scratch.resolve()}"))\n'
        f'(allow file-read* (subpath "{scratch.resolve()}"))\n'
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
