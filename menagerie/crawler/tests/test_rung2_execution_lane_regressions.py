"""Regression tests for the two rung-2 execution-lane failures (m5915, 2026-08-04).

Failure A -- ``observed environment generation is stale or self-attested``: the
driver's strict binding records the LAYERED generation-v2 identity
(``layered_environment_generation(base, content_seal)``) on every attempt, but
the production-layout reducer and the checkpoint validator recomputed only the
v1 base from committed artifacts and demanded raw equality.  Every strict
authority-bound worker attempt was therefore refused at append, which also
swallowed the underlying worker outcome (the m5915 SIGKILL attempt never
reached the ledger).  The fix teaches both validators the layer while keeping
the anchor machine-derived: the declared base must equal the base the
validator itself recomputes from committed bytes, or the record still refuses.

Failure B -- ``worker terminated by signal 9``: the macOS worker memory cap
sampled ``ri_resident_size``, which the kernel holds below any RAM-sized cap
under memory pressure (pages compress/swap) while the true footprint keeps
growing until the OS memorystatus killer SIGKILLs the worker.  The sampler now
reads ``max(ri_resident_size, ri_phys_footprint)`` -- the metric jetsam itself
enforces -- so the configured cap binds first and the breach classifies as
``resource``/``rss-cap`` instead of an anonymous native signal.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.authority import layered_environment_generation
from menagerie.crawler.env_lifecycle import (
    materialized_environment_generation,
    parse_probe_receipt_bytes,
    parse_resolved_export,
)
from menagerie.crawler.envs import DEFAULT_ENVS_ROOT, load_environment_registry
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.reducer import CanonicalReducer, ReductionError, default_ledger_paths
from menagerie.crawler.tests.conftest import make_authority_context, make_failed_attempt
from menagerie.crawler.checkpoint import create_canonical_checkpoint
from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION_V3
from menagerie.crawler.recordio import JsonlLedger
from menagerie.crawler.tests.conftest import make_attempt
from menagerie.crawler.tests.test_checkpoint_transaction import (
    RecordingGit,
    _authority_context,
    _clean_state,
    _write_exact_environment_artifacts,
)
from menagerie.crawler.worker_supervisor import _rusage_v2_memory_bytes

_TARGET = "osx-arm64"
_CONTENT_SEAL = "sha256:" + "b" * 64
_STABLE_ID = "m_example"


def _production_environment_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Materialize the production records/envs layout with exact artifacts.

    Parameters
    ----------
    tmp_path:
        Isolated repository root.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Production records root and copied environment registry root.
    """

    crawler = tmp_path / "menagerie" / "crawler"
    records_root = crawler / "records"
    records_root.mkdir(parents=True)
    envs_root = crawler / "envs"
    shutil.copytree(DEFAULT_ENVS_ROOT, envs_root)
    _write_exact_environment_artifacts(envs_root, _TARGET)
    return records_root, envs_root


def _base_generation(envs_root: Path, attempt: dict[str, Any]) -> str:
    """Recompute the exact v1 base generation the reducer derives itself."""

    intent = load_environment_registry(envs_root, target=_TARGET).intents["core"]
    assert intent.lock.lock_bytes is not None
    assert intent.lock.export_bytes is not None
    package_bytes = parse_resolved_export(intent.lock.export_bytes)
    probe_results = parse_probe_receipt_bytes(
        intent.probes,
        intent.lock.lock_path.with_name(f"{_TARGET}.probes.json").read_bytes(),
    )
    return materialized_environment_generation(
        intent,
        lock_bytes=intent.lock.lock_bytes,
        export_bytes=intent.lock.export_bytes,
        package_bytes=package_bytes,
        python_version=str(attempt["environment"]["python"]),
        compiler_identity=str(attempt["environment"]["compiler_identity"]),
        sdk_identity=str(attempt["environment"]["sdk_identity"]),
        probe_results=probe_results,
    )


def _observed_worker_attempt(envs_root: Path) -> tuple[dict[str, Any], str]:
    """Build one failed worker attempt bound to the committed environment.

    The attempt mirrors the m5915 SIGKILL outcome: stage ``runner``, reason
    ``signal``, with fully observed environment facts.  Returns the attempt and
    the machine-derived v1 base generation for its facts.
    """

    attempt = make_failed_attempt(_STABLE_ID, stage="runner", reason_code="signal")
    intent = load_environment_registry(envs_root, target=_TARGET).intents["core"]
    assert intent.lock.lock_bytes is not None
    assert intent.lock.export_bytes is not None
    package_bytes = parse_resolved_export(intent.lock.export_bytes)
    attempt["environment"] = {
        "family": "core",
        "target": _TARGET,
        "env_id": "env-core",
        "lock_sha256": hash_bytes(intent.lock.lock_bytes),
        "resolved_export_sha256": hash_bytes(intent.lock.export_bytes),
        "python": "3.11",
        "packages_manifest_sha256": hash_bytes(package_bytes),
        "compiler_identity": "test-compiler",
        "sdk_identity": "test-sdk",
        "authority_epoch": None,
        "base_environment_generation": None,
        "environment_content_sha256": None,
        "environment_authority_id": None,
        "selected_interpreter_relative_path": None,
        "selected_interpreter_digest": None,
        "external_escape_records": [],
    }
    attempt["supervisor_observation"].update(
        {
            "stdout_sha256": hash_bytes(b""),
            "stderr_sha256": hash_bytes(b""),
        }
    )
    base = _base_generation(envs_root, attempt)
    attempt["identities"]["environment"] = base
    return attempt, base


def _append(records_root: Path, attempt: dict[str, Any]) -> Any:
    """Append one attempt through a production-layout canonical reducer."""

    context = make_authority_context([_STABLE_ID])
    with CanonicalReducer(default_ledger_paths(records_root), context) as reducer:
        return reducer.append_attempt(attempt)


def test_reducer_accepts_layered_generation_from_strict_binding(tmp_path: Path) -> None:
    """A strict authority-bound attempt carries the v2 layered identity and appends.

    This is the exact m5915 pilot refusal: ``identities.environment`` held
    ``layered(base, content_seal)`` while the reducer recomputed only ``base``
    and raised ``observed environment generation is stale or self-attested``.
    """

    records_root, envs_root = _production_environment_fixture(tmp_path)
    attempt, base = _observed_worker_attempt(envs_root)
    attempt["environment"]["authority_epoch"] = "menagerie.crawler.environment-authority.v1"
    attempt["environment"]["base_environment_generation"] = base
    attempt["environment"]["environment_content_sha256"] = _CONTENT_SEAL
    attempt["identities"]["environment"] = layered_environment_generation(base, _CONTENT_SEAL)
    result = _append(records_root, attempt)
    assert result.appended


def test_reducer_still_accepts_plain_base_generation(tmp_path: Path) -> None:
    """A binding without the authority layer keeps the historical base contract."""

    records_root, envs_root = _production_environment_fixture(tmp_path)
    attempt, _base = _observed_worker_attempt(envs_root)
    result = _append(records_root, attempt)
    assert result.appended


def test_reducer_rejects_self_attested_base_generation(tmp_path: Path) -> None:
    """A layered identity chained to a relabeled base still refuses.

    The layer is only trusted over the base the reducer itself recomputes from
    committed bytes; a driver cannot substitute its own base and launder it
    through a self-consistent layered hash.
    """

    records_root, envs_root = _production_environment_fixture(tmp_path)
    attempt, _base = _observed_worker_attempt(envs_root)
    forged_base = "sha256:" + "c" * 64
    attempt["environment"]["base_environment_generation"] = forged_base
    attempt["environment"]["environment_content_sha256"] = _CONTENT_SEAL
    attempt["identities"]["environment"] = layered_environment_generation(
        forged_base, _CONTENT_SEAL
    )
    with pytest.raises(ReductionError, match="does not chain to the committed base"):
        _append(records_root, attempt)


def test_reducer_rejects_stale_layered_identity(tmp_path: Path) -> None:
    """A layered attempt whose final identity mismatches the layer still refuses."""

    records_root, envs_root = _production_environment_fixture(tmp_path)
    attempt, base = _observed_worker_attempt(envs_root)
    attempt["environment"]["base_environment_generation"] = base
    attempt["environment"]["environment_content_sha256"] = _CONTENT_SEAL
    # The recorded identity ignores the declared content seal: stale/self-attested.
    attempt["identities"]["environment"] = base
    with pytest.raises(ReductionError, match="stale or self-attested"):
        _append(records_root, attempt)


def test_reducer_rejects_half_declared_authority_layer(tmp_path: Path) -> None:
    """A content seal without its base (or vice versa) is malformed and refuses."""

    records_root, envs_root = _production_environment_fixture(tmp_path)
    attempt, base = _observed_worker_attempt(envs_root)
    attempt["environment"]["environment_content_sha256"] = _CONTENT_SEAL
    attempt["identities"]["environment"] = layered_environment_generation(base, _CONTENT_SEAL)
    with pytest.raises(ReductionError, match="does not chain to the committed base"):
        _append(records_root, attempt)


def test_checkpoint_attests_layered_generation(tmp_path: Path) -> None:
    """The checkpoint validator accepts an authority-layered runtime attestation.

    The checkpoint's environment-candidate validation duplicated the reducer's
    v1-only comparison, so a campaign whose every attempt was strict
    authority-bound could never produce a canonical checkpoint: its runtime
    attestations all carry the layered v2 generation.
    """

    snapshot, mirrors = _clean_state(tmp_path)
    crawler_envs = tmp_path / "menagerie" / "crawler" / "envs"
    shutil.copytree(DEFAULT_ENVS_ROOT, crawler_envs)
    lock, export = _write_exact_environment_artifacts(crawler_envs, _TARGET)
    records_root = tmp_path / "menagerie" / "crawler" / "records"

    attempt = make_attempt(snapshot.items[0].stable_id)
    intent = load_environment_registry(crawler_envs, target=_TARGET).intents["core"]
    assert intent.lock.export_bytes is not None
    package_bytes = parse_resolved_export(intent.lock.export_bytes)
    attempt["environment"].update(
        {
            "family": "core",
            "target": _TARGET,
            "lock_sha256": hash_bytes(lock.read_bytes()),
            "resolved_export_sha256": hash_bytes(export.read_bytes()),
            "packages_manifest_sha256": hash_bytes(package_bytes),
            "authority_epoch": "menagerie.crawler.environment-authority.v1",
            "environment_content_sha256": _CONTENT_SEAL,
        }
    )
    base = _base_generation(crawler_envs, attempt)
    attempt["environment"]["base_environment_generation"] = base
    attempt["identities"]["environment"] = layered_environment_generation(base, _CONTENT_SEAL)
    with JsonlLedger(
        records_root / "attempts" / "environment-attestation.jsonl",
        ATTEMPT_SCHEMA_VERSION_V3,
    ) as ledger:
        ledger.append(attempt)

    create_canonical_checkpoint(
        tmp_path,
        snapshot.root,
        mirrors=mirrors,
        authority_context=_authority_context(snapshot),
        branch="menagerie/crawler-pipeline",
        git_runner=RecordingGit(),
    )


def test_darwin_memory_sampler_counts_compressed_footprint() -> None:
    """The macOS sampler must report phys_footprint, not just resident size.

    On the rung-2 host a ~187 GB random-init Mixtral construction kept resident
    size below the 12 GiB cap (the kernel compressed/swapped it under pressure)
    until jetsam SIGKILLed the worker; sampling ``ri_phys_footprint`` makes the
    configured cap bind first.
    """

    buffer = bytearray(256)
    resident = 1 * 1024**3
    footprint = 20 * 1024**3
    buffer[64:72] = resident.to_bytes(8, "little")
    buffer[72:80] = footprint.to_bytes(8, "little")
    assert _rusage_v2_memory_bytes(bytes(buffer)) == footprint


def test_darwin_memory_sampler_is_monotonic_with_resident_reading() -> None:
    """The sampler never reads below the historical resident-only value."""

    buffer = bytearray(256)
    resident = 3 * 1024**3
    buffer[64:72] = resident.to_bytes(8, "little")
    buffer[72:80] = (0).to_bytes(8, "little")
    assert _rusage_v2_memory_bytes(bytes(buffer)) == resident
