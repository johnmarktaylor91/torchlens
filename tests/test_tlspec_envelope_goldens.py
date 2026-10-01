"""Golden-corpus integrity pins for the harvested release artifacts.

The corpus at ``tests/release_goldens/genuine_release_artifacts.tar.gz`` is
harvested BYTES (ecosystem MEMO section 3.7 / build item G1): seven genuine
artifacts from six writers, parts of which are irreplaceable. These tests pin
the exact bytes and the member inventory so silent rot, partial checkouts, or
well-meaning "regeneration" fail loudly.
"""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.smoke]

GOLDENS_DIR = Path(__file__).parent / "release_goldens"
CORPUS_PATH = GOLDENS_DIR / "genuine_release_artifacts.tar.gz"

#: The harvested-corpus digest of record (MEMO section 6 G1). Changing this
#: value is a governance event: goldens are never regenerated, and deletion
#: requires a ledger retirement row.
CORPUS_SHA256 = (
    "3429a84fbd406d374afde5b50f6a8be0867ada77b728d6e88309713c6eadcdb3"  # pragma: allowlist secret
)
CORPUS_SIZE_BYTES = 4_804_520

#: One directory per harvested artifact, exactly as written by its era writer.
CORPUS_MEMBERS = {
    "art_v2.16.0_portable",
    "art_v2.31.0_audit",
    "art_v2.31.0_portable",
    "art_v2.32.4_portable",
    "art_v2.33.0_portable",
    "art_v2.34.1_portable",
    "art_main_portable",
}


def test_corpus_bytes_match_harvest_digest() -> None:
    """The committed tarball is byte-identical to the harvested corpus."""

    data = CORPUS_PATH.read_bytes()
    assert len(data) == CORPUS_SIZE_BYTES
    assert hashlib.sha256(data).hexdigest() == CORPUS_SHA256


def test_corpus_member_inventory() -> None:
    """Every harvested artifact is present; nothing foreign rides along."""

    with tarfile.open(CORPUS_PATH, "r:gz") as tar:
        top_level = {name.split("/", 1)[0] for name in tar.getnames()}
    assert top_level == CORPUS_MEMBERS


def test_provenance_rows_cover_every_member() -> None:
    """PROVENANCE carries a row for each artifact and pins the same digest."""

    provenance = (GOLDENS_DIR / "PROVENANCE").read_text(encoding="utf-8")
    assert CORPUS_SHA256 in provenance
    for member in CORPUS_MEMBERS:
        assert f"artifact: {member}" in provenance


def test_era_generators_are_beside_the_bytes() -> None:
    """The era generator scripts ride with the corpus (provenance, not recipe)."""

    generators = GOLDENS_DIR / "generators"
    for script in ("write_artifact.py", "write216.py"):
        assert (generators / script).is_file()
