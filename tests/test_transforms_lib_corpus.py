"""Corpus fixture integrity + licensing gates (memo B9; lane F19).

The vendored fast path must stay exactly what its manifest records: every
image byte-hashed, every license unambiguous (CC0 / public domain), the
fixture builder deterministic, and the COCO precision path pinned as a
download-on-demand manifest (URL + sha256) that is never fetched here.
"""

from __future__ import annotations

import pytest
import torch

from tests.transforms_corpus.loader import (
    FIXTURE_CROP,
    FIXTURE_ROWS,
    IMAGES_DIR,
    O14_MIN_DISTINCT,
    build_fixture,
    coco_manifest_rows,
    load_vendored,
    manifest_rows,
    verify_vendored_integrity,
)

pytestmark = pytest.mark.smoke

pytest.importorskip("PIL", reason="the corpus loader decodes JPEG via PIL")


def test_vendored_images_match_their_manifest_hashes() -> None:
    """Byte drift in the vendored corpus refuses (sha256 per file)."""

    verify_vendored_integrity()


def test_manifest_carries_unambiguous_licensing_provenance() -> None:
    """Every row: CC0/PD license, source page, artist, retrieval date."""

    rows = manifest_rows()
    assert len(rows) >= O14_MIN_DISTINCT
    files_on_disk = {path.name for path in IMAGES_DIR.glob("*.jpg")}
    assert files_on_disk == {row["file"] for row in rows}
    for row in rows:
        assert row["license"] in ("CC0", "Public domain"), row["file"]
        assert row["source_page"].startswith("https://commons.wikimedia.org/"), row["file"]
        assert row["artist"], row["file"]
        assert row["retrieved"], row["file"]
        assert len(row["sha256"]) == 64


def test_fixture_builder_is_deterministic_and_shaped() -> None:
    """Same seed -> bit-identical fixture; rows/geometry as declared."""

    source = load_vendored()
    first = build_fixture(source)
    second = build_fixture(source)
    assert torch.equal(first, second)
    assert first.shape == (FIXTURE_ROWS, 3, FIXTURE_CROP, FIXTURE_CROP)
    assert first.dtype == torch.float32
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0
    different = build_fixture(source, seed=8)
    assert not torch.equal(first, different)


def test_coco_precision_manifest_is_pinned_not_vendored() -> None:
    """128 rows, sha256-pinned URLs; no COCO bytes live in the repo."""

    rows = coco_manifest_rows()
    assert len(rows) == 128
    for row in rows:
        assert row["url"].startswith("http://images.cocodataset.org/test2017/")
        assert len(row["sha256"]) == 64
        assert row["bytes"] > 0
    assert not any(row["file"] in {p.name for p in IMAGES_DIR.glob("*")} for row in rows)
