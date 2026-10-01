"""Ecosystem runtime: the compat ledger, pair-consistency, and remedies.

Lane F32 row gate: the REMEDY-ACTUALLY-LOADS test (ecosystem MEMO 3.1/G6)
lives here -- every ledger row whose ``bridge_reader`` is the CURRENT
runtime has its goldens actually loaded by this suite on every run, so no
refusal remedy can ever again name a release unverified to read the
artifact it is recommended for (the shipped ">= 2.33" false-remedy defect).
"""

from __future__ import annotations

import json
import shutil
import tarfile
import warnings
from pathlib import Path

import pytest
from _oracle_env import expect_bundle_minor_version_mismatch

import torchlens as tl
from torchlens._io import TorchLensIOError
from torchlens._io.compat_ledger import (
    LEDGER_ROWS,
    SupportClass,
    bridge_reader_remedy,
    governed_stamps_for_writer,
    ledger_rows,
    pair_is_governed,
    ungoverned_pair_error,
)
from torchlens.ecosystem import compat_window

GOLDENS_DIR = Path(__file__).parent / "release_goldens"
CORPUS_PATH = GOLDENS_DIR / "genuine_release_artifacts.tar.gz"


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Extract the harvested corpus once for this module."""

    root = tmp_path_factory.mktemp("eco_corpus")
    with tarfile.open(CORPUS_PATH, "r:gz") as tar:
        tar.extractall(root, filter="data")
    return root


def _quiet_load(path: Path) -> object:
    """Load one artifact with age advisories silenced."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.load(str(path))


# ---------------------------------------------------------------------------
# The ROW GATE: remedy-actually-loads.
# ---------------------------------------------------------------------------


@pytest.mark.heavy
def test_remedy_actually_loads_every_current_bridge_reader_golden(corpus_dir: Path) -> None:
    """Every golden behind a bridge_reader='current' row loads NOW, live.

    This is the executable form of MEMO 3.1's rule that remedies are DERIVED
    from the ledger and every named remedy release actually loads its golden.
    Rows bridged by the current runtime are proven in-process on every run;
    rows bridged by a historical release carry executed-probe evidence
    (asserted below in ``test_historical_bridge_reader_rows_carry_evidence``).
    """

    checked = 0
    for row in LEDGER_ROWS:
        if row.bridge_reader != "current" or not row.golden_ids:
            continue
        for golden_id in row.golden_ids:
            loaded = _quiet_load(corpus_dir / golden_id)
            assert isinstance(loaded, tl.Trace), (row.writer_release, golden_id)
            checked += 1
    assert checked >= 6, "the current-reader golden sweep must cover the harvested corpus"


@pytest.mark.smoke
def test_historical_bridge_reader_rows_carry_evidence() -> None:
    """A remedy may NAME a historical release only with recorded evidence."""

    for row in LEDGER_ROWS:
        if row.bridge_reader in (None, "current"):
            continue
        evidence = row.bridge_reader_evidence.lower()
        assert "probe" in evidence or "verified" in evidence, row.writer_release
        remedy = bridge_reader_remedy(row.artifact_kind, writer_release=row.writer_release)
        assert f"torchlens=={row.bridge_reader}" in remedy
        assert "migrate v1 does not support" in remedy


@pytest.mark.smoke
def test_v216_refusal_names_verified_reader_not_false_range(corpus_dir: Path) -> None:
    """The genuine v2.16 bundle refuses with the 2.17.0 remedy (defect 5 fixed)."""

    with pytest.raises(tl.errors.TorchLensIOError) as excinfo:
        tl.load(str(corpus_dir / "art_v2.16.0_portable"))
    remedy = excinfo.value.fields["remedy"]
    assert "torchlens==2.17.0" in remedy
    assert ">= 2.33" not in remedy
    assert excinfo.value.fields["code"] == "artifact_version_below_floor"


# ---------------------------------------------------------------------------
# Producer pair-consistency (gate G5).
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_lawful_released_artifacts_load(corpus_dir: Path) -> None:
    """Defect 4 fixed: genuine v2.31.0/v2.32.4 artifacts load under main."""

    for name in ("art_v2.31.0_portable", "art_v2.31.0_audit", "art_v2.32.4_portable"):
        loaded = _quiet_load(corpus_dir / name)
        assert isinstance(loaded, tl.Trace), name


@pytest.mark.smoke
def test_governed_windows() -> None:
    """The governed (writer, stamp) relation matches the measured history."""

    assert pair_is_governed("2.31.0", 6)
    assert pair_is_governed("2.32.4", 6)
    assert pair_is_governed("2.34.1", 6)
    assert pair_is_governed("2.34.1", 8)
    assert not pair_is_governed("2.30.0", 6)
    assert not pair_is_governed("2.31.0", 9)
    assert not pair_is_governed("not-a-version", 6)
    assert governed_stamps_for_writer("2.33.0") == frozenset({6})


@pytest.mark.smoke
def test_forged_pair_refuses_typed(corpus_dir: Path, tmp_path: Path) -> None:
    """An ungoverned pair refuses with the stable ledger code."""

    forged = tmp_path / "forged.tlspec"
    shutil.copytree(corpus_dir / "art_v2.31.0_portable", forged)
    manifest_path = forged / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["torchlens_version"] = "2.30.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with expect_bundle_minor_version_mismatch(), pytest.raises(TorchLensIOError) as excinfo:
        tl.load(str(forged))
    assert excinfo.value.fields["code"] == "artifact_producer_pair_ungoverned"
    assert "2.30.0" in str(excinfo.value)


@pytest.mark.smoke
def test_ungoverned_pair_error_payload() -> None:
    """The pair refusal carries structured fields and a ledger-shaped remedy."""

    error = ungoverned_pair_error("2.30.0", 6)
    assert error.fields["code"] == "artifact_producer_pair_ungoverned"
    assert error.fields["governed_tlspec_versions"] == []
    assert "compat_window" in error.fields["remedy"]


# ---------------------------------------------------------------------------
# The report surface (B1).
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_compat_window_report_shape() -> None:
    """compat_window() renders runtime identity, promise status, and rows."""

    report = compat_window()
    assert report.runtime["floor_tlspec_version"] == 6
    assert report.runtime["producer_floor_writer"] == "2.31.0"
    assert len(report.runtime["writer_contract_digest"]) == 64
    assert report.promise["window_status"] == "pending FORK F1 adjudication"
    writers = {row.writer_release for row in report.rows}
    assert {"2.16.0", "2.31.0", "2.32.4", "2.33.0", "2.34.1"} <= writers
    markdown = report.to_markdown()
    assert "compatibility window" in markdown
    assert "2.31.0" in markdown


@pytest.mark.smoke
def test_promised_until_renders_pending_fork_honestly() -> None:
    """No date is invented while FORK F1 is unadjudicated / rows unretired."""

    for row in ledger_rows():
        rendered = row.promised_until()
        if row.support_class is SupportClass.STABLE_PROMISED and row.retired_on is None:
            assert rendered.startswith("active")
        if row.support_class is SupportClass.INTERNAL_DEV:
            assert rendered.startswith("not-promised")
        if row.retired_on is not None and row.support_class is SupportClass.LEGACY_EXCEPTION:
            assert rendered.startswith("not-promised")


@pytest.mark.smoke
def test_dev_build_identity_row_is_generated_not_curated() -> None:
    """The current-runtime row stamps dev-build identity at call time."""

    row = ledger_rows()[-1]
    assert row.exact_release is False
    assert row.build_dirty_state == "unknown"
    assert row.tlspec_version == 9
    assert row.writer_contract_digest is not None


@pytest.mark.smoke
def test_genuine_v216_detection(corpus_dir: Path) -> None:
    """detect_tlspec_format classifies the GENUINE v2.16 spellings (G7)."""

    from torchlens.io import detect_tlspec_format

    assert detect_tlspec_format(corpus_dir / "art_v2.16.0_portable") == "v2.16_modellog_portable"
