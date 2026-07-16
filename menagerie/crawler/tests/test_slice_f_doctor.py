"""Slice F strict doctor preflight tests with injected host probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pytest

from menagerie.crawler.doctor import DoctorConfig, DoctorError, DoctorProbes, run_doctor


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
