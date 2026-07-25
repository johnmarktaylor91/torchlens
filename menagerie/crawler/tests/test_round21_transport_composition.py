"""Round-21 VS7 closed host-transport capability composition proof."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

import menagerie.crawler.policy as policy_module
import menagerie.crawler.worker_supervisor as supervisor_module
from menagerie.crawler.tests import test_anti_substitution_inventories as structural
from menagerie.crawler.tests.conftest import RealEnvironmentFixture
from menagerie.crawler.tests.test_round21_environment_matrix_composition import (
    _adapter_source,
    _assert_award,
    _run_composition,
)


def _unlisted_host_library(transport_members: frozenset[Path]) -> Path:
    """Return an existing host shared library outside the exact transport set.

    Parameters
    ----------
    transport_members:
        Resolved interpreter ELF dependencies that form the closed positive set.

    Returns
    -------
    pathlib.Path
        Resolved existing ``/usr/lib`` shared library excluded from the capability.
    """

    root = Path("/usr/lib")
    for candidate in sorted(root.rglob("*.so*"), key=lambda path: str(path)):
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file() and resolved not in transport_members:
            return resolved
    raise AssertionError("Linux P07 requires an unlisted existing /usr/lib shared library")


def _successful_worker_transport_reads(
    root: Path,
    fixture: RealEnvironmentFixture,
    capability: policy_module.HostTransportLibraryCapability,
) -> frozenset[Path]:
    """Return successful outside-prefix shared-library opens from parent telemetry.

    Parameters
    ----------
    root:
        Real-composition root containing parent-owned strace logs.
    fixture:
        Selected environment whose sealed prefix is ordinary runtime authority.
    capability:
        Exact host transport capability expected to cover the remaining libraries.

    Returns
    -------
    frozenset[pathlib.Path]
        Canonical successful worker library reads outside the sealed prefix.
    """

    audit_paths = tuple(root.rglob("sandbox-syscalls.log"))
    if not audit_paths:
        raise AssertionError("P07 real composition produced no parent strace telemetry")
    observed: set[Path] = set()
    for audit_path in audit_paths:
        records = supervisor_module._complete_trace_records(  # noqa: SLF001
            audit_path.read_text(encoding="utf-8").splitlines()
        )
        if records is None:
            raise AssertionError("P07 parent strace telemetry is incomplete")
        worker_records = supervisor_module._worker_trace_records(  # noqa: SLF001
            records,
            capability,
        )
        if worker_records is None:
            raise AssertionError("P07 parent strace telemetry lacks the worker exec")
        for line in worker_records:
            if supervisor_module._syscall_name(line) not in {  # noqa: SLF001
                "open",
                "openat",
                "openat2",
            }:
                continue
            result = supervisor_module._read_only_open_result(line)  # noqa: SLF001
            paths = supervisor_module._decoded_trace_paths(line)  # noqa: SLF001
            if result is None or result < 0 or not paths:
                continue
            candidate = Path(paths[0]).resolve()
            if candidate.is_relative_to(fixture.prefix):
                continue
            if candidate.suffix == ".so" or ".so." in candidate.name.lower():
                observed.add(candidate)
    return frozenset(observed)


def test_round21_closed_transport_capability_awards_and_rejects_unlisted_library(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """Exact interpreter transport reads award while an unlisted host library is rejected.

    Parameters
    ----------
    tmp_path:
        Isolated real-composition campaign root.
    real_environment_fixture:
        Strictly bound lock-selected hardlink clone used by the shipped compiler.
    """

    if sys.platform != "linux":
        message = "Linux P07 host is unavailable"
        if os.environ.get("MENAGERIE_RELEASE_GATE") == "1":
            pytest.fail(f"unmet-release-gate: {message}")
        pytest.skip(message)
    sandbox = policy_module.detect_os_sandbox()
    if sandbox is None or sandbox.kind != "bubblewrap":
        message = "bubblewrap P07 enforcement is unavailable"
        if os.environ.get("MENAGERIE_RELEASE_GATE") == "1":
            pytest.fail(f"unmet-release-gate: {message}")
        pytest.skip(message)

    interpreter = real_environment_fixture.binding.python_executable
    capability = policy_module._linux_host_transport_library_capability(  # noqa: SLF001
        interpreter
    )
    transport_members = frozenset(capability.canonical_members)
    assert transport_members
    assert capability.interpreter == interpreter.resolve()
    assert capability.digest.startswith("sha256:")
    mounts = policy_module._linux_minimal_read_mounts(  # noqa: SLF001
        (str(interpreter),),
        Path.cwd(),
        (),
        host_transport_capability=capability,
    )
    assert all(
        path in mounts or path.is_relative_to(real_environment_fixture.prefix)
        for path in capability.members
    )
    assert all(
        supervisor_module._system_transport_library_path_allowed(  # noqa: SLF001
            path,
            capability,
        )
        for path in transport_members
    )
    unlisted_library = _unlisted_host_library(transport_members)
    assert not supervisor_module._system_transport_library_path_allowed(  # noqa: SLF001
        unlisted_library,
        capability,
    )
    assert not supervisor_module._system_transport_library_path_allowed(  # noqa: SLF001
        unlisted_library
    )

    observation = _run_composition(
        tmp_path / "award",
        real_environment_fixture,
        _adapter_source(
            "assert os.environ.get('MENAGERIE_ROUND19_PTH_SENTINEL') == 'sealed-startup'"
        ),
    )
    assert observation.attempts[0]["result"] == "succeeded", observation.attempts[0][
        "policy_observation"
    ]
    _assert_award(observation)
    observed_transport_reads = _successful_worker_transport_reads(
        tmp_path / "award",
        real_environment_fixture,
        capability,
    )
    assert observed_transport_reads
    assert observed_transport_reads <= transport_members
    assert all(
        supervisor_module._system_transport_library_path_allowed(  # noqa: SLF001
            path,
            capability,
        )
        for path in observed_transport_reads
    )
    assert set(structural.ROUND21_VS7_PROOF_REGISTRY) == {
        "P01",
        "P02",
        "P03",
        "P04",
        "P05",
        "P06",
        "P07",
        "P12",
        "P13",
        "P14",
        "P17",
        "P19",
        "T01",
        "T01-CI",
        "T02",
        "T03",
    }
