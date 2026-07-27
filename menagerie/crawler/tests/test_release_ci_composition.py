"""Round-21 committed-lock and release-CI real compositions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest

from menagerie.crawler.env_lifecycle import (
    parse_exact_lock,
    parse_probe_receipt_bytes,
    parse_resolved_export,
)
from menagerie.crawler.envs import ExportCheck, IntentProbes
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.tests.test_environment_authority_composition import (
    _run_host_denial_composition,
    test_macos_v3_profile_has_one_fresh_literal_prefix_and_exact_outside_members,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SPEC_PATH = _REPOSITORY_ROOT / "menagerie/crawler/envs/specs/round21-release.yml"
_PROBE_CONTRACT_PATH = _SPEC_PATH.with_name("round21-release.probes.json")
_LOCK_PATH = _REPOSITORY_ROOT / "menagerie/crawler/envs/locks/round19-linux-64.lock"
_LOCK_FAMILY = {
    "lock": _LOCK_PATH,
    "resolved_export": _LOCK_PATH.with_suffix(".resolved.json"),
    "resolved_export_hash": _LOCK_PATH.with_suffix(".resolved.sha256"),
    "provenance": _LOCK_PATH.with_suffix(".provenance.json"),
    "probe_receipt": _LOCK_PATH.with_suffix(".probes.json"),
}
_MACOS_LOCK_PATH = _REPOSITORY_ROOT / "menagerie/crawler/envs/locks/round19-osx-arm64.lock"
_MACOS_LOCK_FAMILY = {
    "lock": _MACOS_LOCK_PATH,
    "resolved_export": _MACOS_LOCK_PATH.with_suffix(".resolved.json"),
    "resolved_export_hash": _MACOS_LOCK_PATH.with_suffix(".resolved.sha256"),
    "provenance": _MACOS_LOCK_PATH.with_suffix(".provenance.json"),
    "probe_receipt": _MACOS_LOCK_PATH.with_suffix(".probes.json"),
    "virtual_package_spec": _SPEC_PATH.with_name("round21-release.virtual-packages.yml"),
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Require a string-keyed mapping.

    Parameters
    ----------
    value:
        Candidate decoded JSON value.
    label:
        Diagnostic label.

    Returns
    -------
    collections.abc.Mapping[str, Any]
        Validated mapping.
    """

    assert isinstance(value, dict), f"{label} must be a mapping"
    assert all(isinstance(key, str) for key in value), f"{label} has a non-string key"
    return value


def _probe_contract() -> IntentProbes:
    """Load the committed target-neutral release probe contract.

    Returns
    -------
    IntentProbes
        Exact import and export probes shared by supported hosts.
    """

    raw = _mapping(json.loads(_PROBE_CONTRACT_PATH.read_bytes()), "probe contract")
    assert set(raw) == {"schema_version", "imports", "export_checks", "source_build"}
    assert raw["schema_version"] == "menagerie.crawler.release-probes.v1"
    imports = raw["imports"]
    export_checks = raw["export_checks"]
    assert isinstance(imports, list) and len(imports) >= 3
    assert all(isinstance(value, str) and value for value in imports)
    assert isinstance(export_checks, list)
    checks = tuple(
        ExportCheck(
            module=str(_mapping(value, "export check")["module"]),
            attribute=str(_mapping(value, "export check")["attribute"]),
        )
        for value in export_checks
    )
    assert raw["source_build"] == []
    return IntentProbes(tuple(imports), checks, ())


@pytest.mark.skipif(
    os.environ.get("MENAGERIE_RELEASE_GATE") != "1",
    reason="release-gate-only",
)
@pytest.mark.round21_linux_real
def test_linux_committed_lock_provenance_awards_in_ci(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """A committed Linux lock must strict-bind, award, and catch a denied read."""

    if sys.platform != "linux":
        pytest.skip("Linux release host is unavailable")
    assert os.environ.get("MENAGERIE_RELEASE_GATE") == "1"
    assert {name for name, path in _LOCK_FAMILY.items() if path.is_file()} == set(_LOCK_FAMILY)
    assert _SPEC_PATH.is_file()
    assert _PROBE_CONTRACT_PATH.is_file()
    selected_lock = Path(os.environ["MENAGERIE_PLATFORM_LOCK"]).resolve()
    assert selected_lock == _LOCK_PATH.resolve()

    lock_bytes = _LOCK_FAMILY["lock"].read_bytes()
    export_bytes = _LOCK_FAMILY["resolved_export"].read_bytes()
    assert parse_exact_lock(lock_bytes)
    assert parse_resolved_export(export_bytes) == export_bytes
    declared_export_hash = _LOCK_FAMILY["resolved_export_hash"].read_text(encoding="utf-8").strip()
    assert declared_export_hash == hash_bytes(export_bytes)
    probes = _probe_contract()
    probe_results = parse_probe_receipt_bytes(probes, _LOCK_FAMILY["probe_receipt"].read_bytes())

    provenance = _mapping(
        json.loads(_LOCK_FAMILY["provenance"].read_bytes()), "Linux lock provenance"
    )
    assert provenance["schema_version"] == "menagerie.crawler.release-lock-provenance.v1"
    assert provenance["target"] == "linux-64"
    assert provenance["spec_path"] == _SPEC_PATH.relative_to(_REPOSITORY_ROOT).as_posix()
    assert provenance["spec_sha256"] == hash_bytes(_SPEC_PATH.read_bytes())
    assert provenance["lock_sha256"] == hash_bytes(lock_bytes)
    assert provenance["resolved_export_sha256"] == declared_export_hash
    assert provenance["probe_contract_sha256"] == hash_bytes(_PROBE_CONTRACT_PATH.read_bytes())
    assert provenance["probe_receipt_sha256"] == hash_bytes(
        _LOCK_FAMILY["probe_receipt"].read_bytes()
    )
    assert provenance["clean_create"]["validated"] is True

    fixture = request.getfixturevalue("real_environment_fixture")
    assert fixture.intent.lock.lock_path.resolve() == _LOCK_PATH.resolve()
    assert fixture.intent.lock.lock_bytes == lock_bytes
    assert fixture.intent.lock.export_bytes == export_bytes
    assert fixture.intent.probes == probes
    assert fixture.probe_results == probe_results
    assert fixture.binding.python_executable == fixture.prefix / "bin/python"
    assert fixture.binding.python_executable.resolve() != Path(sys.executable).resolve()
    _run_host_denial_composition(tmp_path, fixture, expected_sandbox="bubblewrap")


@pytest.mark.round21_macos_real
def test_macos_committed_lock_seatbelt_award_and_denial(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """A committed macOS lock must strict-bind, award, and audit a Seatbelt denial."""

    if sys.platform != "darwin" or os.uname().machine != "arm64":
        pytest.skip("hosted Darwin arm64 release host is unavailable")
    assert os.environ.get("MENAGERIE_RELEASE_GATE") == "1"
    assert {name for name, path in _MACOS_LOCK_FAMILY.items() if path.is_file()} == set(
        _MACOS_LOCK_FAMILY
    )
    selected_lock = Path(os.environ["MENAGERIE_PLATFORM_LOCK"]).resolve()
    assert selected_lock == _MACOS_LOCK_PATH.resolve()

    lock_bytes = _MACOS_LOCK_FAMILY["lock"].read_bytes()
    export_bytes = _MACOS_LOCK_FAMILY["resolved_export"].read_bytes()
    assert parse_exact_lock(lock_bytes)
    assert parse_resolved_export(export_bytes) == export_bytes
    declared_export_hash = (
        _MACOS_LOCK_FAMILY["resolved_export_hash"].read_text(encoding="utf-8").strip()
    )
    assert declared_export_hash == hash_bytes(export_bytes)
    probes = _probe_contract()
    probe_results = parse_probe_receipt_bytes(
        probes, _MACOS_LOCK_FAMILY["probe_receipt"].read_bytes()
    )

    provenance = _mapping(
        json.loads(_MACOS_LOCK_FAMILY["provenance"].read_bytes()), "macOS lock provenance"
    )
    assert provenance["schema_version"] == "menagerie.crawler.release-lock-provenance.v1"
    assert provenance["target"] == "osx-arm64"
    assert provenance["spec_path"] == _SPEC_PATH.relative_to(_REPOSITORY_ROOT).as_posix()
    assert provenance["spec_sha256"] == hash_bytes(_SPEC_PATH.read_bytes())
    assert provenance["lock_sha256"] == hash_bytes(lock_bytes)
    assert provenance["resolved_export_sha256"] == declared_export_hash
    assert provenance["probe_contract_sha256"] == hash_bytes(_PROBE_CONTRACT_PATH.read_bytes())
    assert provenance["probe_receipt_sha256"] == hash_bytes(
        _MACOS_LOCK_FAMILY["probe_receipt"].read_bytes()
    )
    assert provenance["virtual_package_spec_sha256"] == hash_bytes(
        _MACOS_LOCK_FAMILY["virtual_package_spec"].read_bytes()
    )
    assert provenance["solver"]["name"] == "conda-lock"
    assert "osx-arm64" in provenance["solver"]["command"]
    assert provenance["probe_observation"] == {
        "committed_on_linux": False,
        "producer": "hosted macOS release CI attestation",
    }
    assert provenance["clean_create"]["validation_host"] == "hosted macOS release CI"

    fixture = request.getfixturevalue("real_environment_fixture")
    assert fixture.intent.lock.lock_path.resolve() == _MACOS_LOCK_PATH.resolve()
    assert fixture.intent.lock.lock_bytes == lock_bytes
    assert fixture.intent.lock.export_bytes == export_bytes
    assert fixture.intent.probes == probes
    assert fixture.probe_results == probe_results
    assert fixture.binding.python_executable == fixture.prefix / "bin/python"
    assert fixture.binding.python_executable.resolve() != Path(sys.executable).resolve()
    test_macos_v3_profile_has_one_fresh_literal_prefix_and_exact_outside_members(tmp_path)
    _run_host_denial_composition(tmp_path, fixture, expected_sandbox="sandbox-exec")
