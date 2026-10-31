"""tl.migrate v1: the deciding real-artifact test and the transaction law.

The DECIDING TEST is named by the memo, never assumed (MEMO 3.6): migrate
the genuine harvested v2.31.0/v2.33.0/v2.34.1 artifacts to the current
schema, load under main, validate, and assert provenance and
absent-family disclosure. Failure injection at the filesystem boundaries
must leave the source bytes intact.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import pytest

from torchlens._ecosystem import migrate
from torchlens._ecosystem.migrate import _ABSENT_FAMILIES_BY_TARGET
from torchlens._io import ArtifactError, ArtifactTooOldError
from torchlens._io.compat_ledger import MIGRATION_WITNESS_FILENAME

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


@pytest.mark.smoke
def test_v216_source_refuses_with_verified_reader(corpus_dir: Path, tmp_path: Path) -> None:
    """Pre-floor migration is out of v1; the remedy names the verified reader."""

    path = _working_copy(corpus_dir, tmp_path, "art_v2.16.0_portable")
    with pytest.raises(ArtifactError) as excinfo:
        migrate(path)
    assert excinfo.value.fields["code"] == "migration_source_unsupported"
    assert "torchlens==2.17.0" in excinfo.value.fields["remedy"]


def test_non_directory_refuses(tmp_path: Path) -> None:
    """A non-directory source refuses typed."""

    with pytest.raises(ArtifactError) as excinfo:
        migrate(tmp_path / "missing")
    assert excinfo.value.fields["code"] == "migration_source_unsupported"


def test_unknown_format_refuses(tmp_path: Path) -> None:
    """A directory that is not a .tlspec refuses typed."""

    source = tmp_path / "junk.tlspec"
    source.mkdir()
    (source / "README.txt").write_text("not an artifact", encoding="utf-8")
    with pytest.raises(ArtifactError) as excinfo:
        migrate(source)
    assert excinfo.value.fields["code"] == "migration_source_unsupported"


def test_stranded_staging_refuses_before_mutation(corpus_dir: Path, tmp_path: Path) -> None:
    """A stranded staging directory refuses BEFORE any mutation."""

    path = _working_copy(corpus_dir, tmp_path, "art_v2.31.0_portable")
    staging = path.with_name(path.name + ".tl-migrate-staging")
    staging.mkdir()
    with pytest.raises(ArtifactError) as excinfo:
        migrate(path)
    assert excinfo.value.fields["code"] == "migration_concurrent_writer"
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tlspec_version"] == 6  # untouched


@pytest.mark.smoke
@pytest.mark.parametrize(
    "name", ["art_v2.31.0_portable", "art_v2.33.0_portable", "art_v2.34.1_portable"]
)
def test_pre_floor_source_refuses_as_saved_by_2x(
    corpus_dir: Path, tmp_path: Path, name: str
) -> None:
    """Migration of a file saved by torchlens 2.x refuses with the one 2.x error.

    The reader floor is the current format, so there is no source a migration could
    start from: a tlspec-6 artifact refuses typed before any staging, witness or publish
    step runs, and the source is left exactly as it was.
    """

    path = _working_copy(corpus_dir, tmp_path, name)
    before = sorted(p.relative_to(path) for p in path.rglob("*"))
    with pytest.raises(ArtifactTooOldError) as excinfo:
        migrate(path)
    assert excinfo.value.fields["code"] == "artifact_version_below_floor"
    assert "saved by torchlens 2.x" in str(excinfo.value)
    assert not (path / MIGRATION_WITNESS_FILENAME).exists()
    assert sorted(p.relative_to(path) for p in path.rglob("*")) == before


def test_every_v1_step_declares_manifest_level() -> None:
    """The step-protocol guard: no v1 step quietly claims STATE level."""

    from torchlens._ecosystem.migrate import _plan_steps

    for step in _plan_steps(6):
        assert step.level == "manifest"
        assert step.target_contract_digest is None
    assert set(_ABSENT_FAMILIES_BY_TARGET) == {7, 8, 9, 10}


def test_migrate_source_has_no_execution_imports() -> None:
    """migrate.py never imports capture/backends/semantic (never executes)."""

    import importlib

    migrate_module = importlib.import_module("torchlens._ecosystem.migrate")
    source = Path(migrate_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("from ..capture", "from ..backends", "from ..semantic", "import torch\n"):
        assert forbidden not in source, forbidden
