"""tl.migrate v1: the deciding real-artifact test and the transaction law.

The DECIDING TEST is named by the memo, never assumed (MEMO 3.6): migrate
the genuine harvested v2.31.0/v2.33.0/v2.34.1 artifacts to the current
schema, load under main, validate, and assert provenance and
absent-family disclosure. Failure injection at the filesystem boundaries
must leave the source bytes intact.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import warnings
from pathlib import Path

import pytest
from _oracle_env import expect_bundle_minor_version_mismatch

import torchlens as tl
from torchlens._io import TorchLensIOError
from torchlens._io.compat_ledger import MIGRATION_WITNESS_FILENAME, read_migration_witness
from torchlens.ecosystem import migrate
from torchlens.ecosystem.migrate import _ABSENT_FAMILIES_BY_TARGET

pytestmark = pytest.mark.smoke

GOLDENS_DIR = Path(__file__).parent / "release_goldens"
CORPUS_PATH = GOLDENS_DIR / "genuine_release_artifacts.tar.gz"


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Extract the harvested corpus once for this module."""

    root = tmp_path_factory.mktemp("eco_migrate_corpus")
    with tarfile.open(CORPUS_PATH, "r:gz") as tar:
        tar.extractall(root, filter="data")
    return root


def _working_copy(corpus_dir: Path, tmp_path: Path, name: str) -> Path:
    """Copy one golden into a writable scratch directory."""

    target = tmp_path / name
    shutil.copytree(corpus_dir / name, target)
    return target


@pytest.mark.parametrize(
    "name", ["art_v2.31.0_portable", "art_v2.33.0_portable", "art_v2.34.1_portable"]
)
def test_deciding_real_artifact_migration(corpus_dir: Path, tmp_path: Path, name: str) -> None:
    """Genuine released artifacts migrate 6 -> current, reload, and validate."""

    from torchlens.validation import validate_tlspec

    path = _working_copy(corpus_dir, tmp_path, name)
    report = migrate(path)
    assert report.status == "migrated"
    assert report.source_tlspec == 6
    assert report.final_tlspec == 9
    assert [step.step_id for step in report.steps] == [
        "tlspec_6_to_7",
        "tlspec_7_to_8",
        "tlspec_8_to_9",
    ]
    assert all(step.level == "manifest" for step in report.steps)
    assert report.backup_path is not None and report.backup_path.is_dir()
    # The migrated artifact loads under main and validates.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = tl.load(str(path))
    assert isinstance(loaded, tl.Trace)
    assert len(loaded.ops) > 0
    validate_tlspec(path)
    # Provenance: witness present, origin preserved, disclosures honest.
    witness = read_migration_witness(path)
    assert witness is not None
    assert witness["origin"]["tlspec_version"] == 6
    assert witness["disclosures"]["replay_attestations"] == "unavailable"
    assert witness["disclosures"]["facts_backfilled"] == "none"
    families = {family for step in witness["steps"] for family in step["absent_fact_families"]}
    assert "capture_outcome" in families and "site_key" in families
    # Writer identity was never forged.
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["torchlens_version"] == name.split("_")[1].lstrip("v")
    assert manifest["tlspec_version"] == 9
    # The retained backup is byte-compatible with the original manifest.
    backup_manifest = json.loads((report.backup_path / "manifest.json").read_text(encoding="utf-8"))
    assert backup_manifest["tlspec_version"] == 6


def test_migration_is_idempotent(corpus_dir: Path, tmp_path: Path) -> None:
    """A second migrate() is a no-op that reports already-current."""

    path = _working_copy(corpus_dir, tmp_path, "art_v2.31.0_portable")
    first = migrate(path)
    assert first.status == "migrated"
    second = migrate(path)
    assert second.status == "already-current"
    assert second.steps == ()
    assert second.witness_path is not None


def test_v216_source_refuses_with_verified_reader(corpus_dir: Path, tmp_path: Path) -> None:
    """Pre-floor migration is out of v1; the remedy names the verified reader."""

    path = _working_copy(corpus_dir, tmp_path, "art_v2.16.0_portable")
    with pytest.raises(TorchLensIOError) as excinfo:
        migrate(path)
    assert excinfo.value.fields["code"] == "migration_source_unsupported"
    assert "torchlens==2.17.0" in excinfo.value.fields["remedy"]


def test_non_directory_refuses(tmp_path: Path) -> None:
    """A non-directory source refuses typed."""

    with pytest.raises(TorchLensIOError) as excinfo:
        migrate(tmp_path / "missing")
    assert excinfo.value.fields["code"] == "migration_source_unsupported"


def test_unknown_format_refuses(tmp_path: Path) -> None:
    """A directory that is not a .tlspec refuses typed."""

    source = tmp_path / "junk.tlspec"
    source.mkdir()
    (source / "README.txt").write_text("not an artifact", encoding="utf-8")
    with pytest.raises(TorchLensIOError) as excinfo:
        migrate(source)
    assert excinfo.value.fields["code"] == "migration_source_unsupported"


def test_stranded_staging_refuses_before_mutation(corpus_dir: Path, tmp_path: Path) -> None:
    """A stranded staging directory refuses BEFORE any mutation."""

    path = _working_copy(corpus_dir, tmp_path, "art_v2.31.0_portable")
    staging = path.with_name(path.name + ".tl-migrate-staging")
    staging.mkdir()
    with pytest.raises(TorchLensIOError) as excinfo:
        migrate(path)
    assert excinfo.value.fields["code"] == "migration_concurrent_writer"
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tlspec_version"] == 6  # untouched


def test_failure_injection_leaves_source_intact(
    corpus_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staging-boundary failure refuses typed and never mutates the source."""

    path = _working_copy(corpus_dir, tmp_path, "art_v2.31.0_portable")
    original = (path / "manifest.json").read_bytes()

    def _boom(src: object, dst: object, **kwargs: object) -> None:
        raise OSError("injected copy failure")

    monkeypatch.setattr(shutil, "copytree", _boom)
    with pytest.raises(TorchLensIOError) as excinfo:
        migrate(path)
    assert excinfo.value.fields["code"] == "migration_staging_failed"
    assert (path / "manifest.json").read_bytes() == original
    assert not path.with_name(path.name + ".tl-migrate-staging").exists()


def test_publish_failure_restores_source(
    corpus_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A final-rename failure restores the source in place, typed."""

    path = _working_copy(corpus_dir, tmp_path, "art_v2.31.0_portable")
    real_rename = os.rename
    calls = {"n": 0}

    def _second_rename_fails(src: object, dst: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("injected publish failure")
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", _second_rename_fails)
    with pytest.raises(TorchLensIOError) as excinfo:
        migrate(path)
    assert excinfo.value.fields["code"] == "migration_staging_failed"
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tlspec_version"] == 6  # restored


def test_tampered_witness_refuses_at_load(corpus_dir: Path, tmp_path: Path) -> None:
    """A witness beside a mismatching manifest is a typed refusal, not a gap."""

    path = _working_copy(corpus_dir, tmp_path, "art_v2.31.0_portable")
    migrate(path)
    witness_path = path / MIGRATION_WITNESS_FILENAME
    witness = json.loads(witness_path.read_text(encoding="utf-8"))
    witness["final_manifest_sha256"] = "0" * 64
    witness_path.write_text(json.dumps(witness), encoding="utf-8")
    with expect_bundle_minor_version_mismatch(), pytest.raises(TorchLensIOError) as excinfo:
        tl.load(str(path))
    # The digest cross-check fails the witness door, so the pair refuses.
    assert excinfo.value.fields["code"] == "artifact_producer_pair_ungoverned"


def test_broken_witness_refuses_typed(corpus_dir: Path, tmp_path: Path) -> None:
    """A structurally broken witness refuses migration_witness_invalid."""

    path = _working_copy(corpus_dir, tmp_path, "art_v2.31.0_portable")
    (path / MIGRATION_WITNESS_FILENAME).write_text('{"witness_schema": "bogus"}', encoding="utf-8")
    with pytest.raises(TorchLensIOError) as excinfo:
        migrate(path)
    assert excinfo.value.fields["code"] == "migration_witness_invalid"


def test_every_v1_step_declares_manifest_level() -> None:
    """The step-protocol guard: no v1 step quietly claims STATE level."""

    from torchlens.ecosystem.migrate import _plan_steps

    for step in _plan_steps(6):
        assert step.level == "manifest"
        assert step.target_contract_digest is None
    assert set(_ABSENT_FAMILIES_BY_TARGET) == {7, 8, 9}


def test_migrate_source_has_no_execution_imports() -> None:
    """migrate.py never imports capture/backends/semantic (never executes)."""

    import importlib

    migrate_module = importlib.import_module("torchlens.ecosystem.migrate")
    source = Path(migrate_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("from ..capture", "from ..backends", "from ..semantic", "import torch\n"):
        assert forbidden not in source, forbidden
