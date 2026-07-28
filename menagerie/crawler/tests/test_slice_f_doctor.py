"""Slice F strict doctor preflight tests with injected host probes."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
from typing import Mapping

import pytest

from menagerie.crawler.doctor import (
    DoctorConfig,
    DoctorError,
    DoctorProbes,
    SystemDoctorProbes,
    run_doctor,
)


@dataclass
class FakeDoctorProbes(DoctorProbes):
    """Fully deterministic doctor environment."""

    system: str = "Darwin"
    architecture: str = "arm64"
    branch_name: str = "menagerie/crawler-pipeline"
    free_bytes: int = 200 * 1024**3
    lock_free: bool = True
    mirrors_ok: bool = True
    tools: frozenset[str] = frozenset({"WebSearch", "web_search_exa", "web_fetch_exa"})
    secrets: tuple[str, ...] = ()
    tripwires: dict[str, bool] = field(
        default_factory=lambda: {"offline": True, "socket": True, "write-audit": True}
    )
    wakeup_ok: bool = True
    violations: tuple[str, ...] = ()
    wrappers: dict[str, str] = field(
        default_factory=lambda: {
            "author": "author 1",
            "checker": "checker 1",
            "environment": "environment 1",
        }
    )
    codex_ok: bool = True
    environment_versions: dict[str, str] = field(
        default_factory=lambda: {
            "conda": "conda 1",
            "mamba": "mamba 1",
            "conda-lock": "conda-lock 1",
        }
    )
    notifier_ok: bool = True
    worker_slot_free: bool = True
    reserve_bytes: int = 200 * 1024**3

    def machine(self) -> tuple[str, str]:
        """Return the configured system and architecture."""

        return (self.system, self.architecture)

    def branch(self) -> str:
        """Return the configured branch."""

        return self.branch_name

    def disk_free_bytes(self) -> int:
        """Return configured disk capacity."""

        return self.free_bytes

    def lock_available(self) -> bool:
        """Return configured lock availability."""

        return self.lock_free

    def mirrors_reachable(self) -> bool:
        """Return configured mirror health."""

        return self.mirrors_ok

    def author_tools(self) -> frozenset[str]:
        """Return configured author tools."""

        return self.tools

    def wrapper_versions(self) -> Mapping[str, str]:
        """Return configured operator wrapper versions."""

        return self.wrappers

    def codex_ready(self) -> bool:
        """Return configured Codex capability status."""

        return self.codex_ok

    def environment_tool_versions(self) -> Mapping[str, str]:
        """Return configured environment tool versions."""

        return self.environment_versions

    def notifier_delivery(self) -> bool:
        """Return configured nonce-delivery status."""

        return self.notifier_ok

    def worker_slot_available(self) -> bool:
        """Return configured global worker-slot availability."""

        return self.worker_slot_free

    def dynamic_disk_reserve_bytes(self) -> int:
        """Return configured dynamic disk reserve."""

        return self.reserve_bytes

    def secret_findings(self) -> tuple[str, ...]:
        """Return configured secret findings."""

        return self.secrets

    def policy_tripwires(self) -> Mapping[str, bool]:
        """Return configured policy self-tests."""

        return self.tripwires

    def wakeup_available(self) -> bool:
        """Return configured wakeup support."""

        return self.wakeup_ok

    def torchlens_import_violations(self) -> tuple[str, ...]:
        """Return configured static import violations."""

        return self.violations


def _config(tmp_path: Path) -> DoctorConfig:
    """Return a strict osx-arm64 doctor configuration."""

    return DoctorConfig(tmp_path, tmp_path / ".crawl-local", "osx-arm64")


def test_doctor_passes_clean_preflight(tmp_path: Path) -> None:
    """Every required clean observation produces a go decision."""

    report = run_doctor(_config(tmp_path), FakeDoctorProbes())
    assert report.passed
    assert set(report.checks.values()) == {"pass"}


@pytest.mark.parametrize(
    ("field_name", "value", "finding"),
    [
        ("branch_name", "main", "branch"),
        ("free_bytes", 1, "disk"),
        ("tools", frozenset({"WebSearch"}), "author-web-tools"),
        (
            "tripwires",
            {"offline": True, "socket": False, "write-audit": True},
            "policy:socket",
        ),
        ("wakeup_ok", False, "wakeup"),
        ("violations", ("adapters/bad.py",), "torchlens-import-ban"),
    ],
)
def test_doctor_fails_critical_preflight_findings(
    tmp_path: Path, field_name: str, value: object, finding: str
) -> None:
    """Every critical branch/disk/tool/policy/import failure is loud and typed."""

    probes = FakeDoctorProbes()
    setattr(probes, field_name, value)
    with pytest.raises(DoctorError) as captured:
        run_doctor(_config(tmp_path), probes)
    assert any(failure.startswith(finding) for failure in captured.value.failures)


def test_doctor_survey_runs_all_checks_without_raising(tmp_path: Path) -> None:
    """Non-strict survey reports mandatory failures honestly and returns."""

    probes = FakeDoctorProbes(branch_name="main", tools=frozenset(), notifier_ok=False)
    report = run_doctor(
        DoctorConfig(tmp_path, tmp_path / ".crawl-local", "osx-arm64", strict=False),
        probes,
    )

    assert not report.passed
    assert report.checks["branch"].startswith("fail:")
    assert report.checks["author-web-tools"].startswith("fail:")
    assert report.checks["notifier-delivery"].startswith("fail:")
    assert report.checks["torchlens-import-ban"] == "pass"


def test_author_tools_requires_fresh_nonce_bound_live_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real probe accepts exercised receipts rather than MCP help text."""

    wrapper = tmp_path / "author-wrapper"
    wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.setenv("MENAGERIE_AUTHOR_COMMAND", str(wrapper))

    def runner(
        argv: list[str] | tuple[str, ...], _cwd: Path, timeout: float = 180.0
    ) -> subprocess.CompletedProcess[str]:
        """Materialize the exact nonce-bound capability receipt."""

        request = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
        receipt = {
            "nonce": request["nonce"],
            "completed_at": request["requested_at"],
            "receipts": [
                {
                    "tool": tool,
                    "nonce": request["nonce"],
                    "exercised": True,
                    "receipt": f"{tool}-receipt",
                }
                for tool in request["required_tools"]
            ],
        }
        Path(request["required_output_path"]).write_text(json.dumps(receipt), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")

    config = DoctorConfig(tmp_path, tmp_path / ".crawl-local", "osx-arm64")
    probes = SystemDoctorProbes(config, command_runner=runner)

    assert probes.author_tools() == frozenset({"WebSearch", "web_search_exa", "web_fetch_exa"})
