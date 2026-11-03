"""Corpus fixture integrity + licensing gates (memo B9; lane F19).

The vendored fast path must stay exactly what its manifest records: every
image byte-hashed, every license unambiguous (CC0 / public domain), the
fixture builder deterministic, and the COCO precision path pinned as a
download-on-demand manifest (URL + sha256). The fetch path is exercised
against a local ``file://`` fixture; the one real fetch is opt-in (``rare``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from tests.transforms_corpus.loader import (
    COCO_URL_PREFIX,
    FIXTURE_CROP,
    FIXTURE_ROWS,
    IMAGES_DIR,
    O14_MIN_DISTINCT,
    CocoChecksumMismatch,
    build_fixture,
    coco_manifest_rows,
    fetch_coco,
    load_vendored,
    manifest_rows,
    verify_vendored_integrity,
)

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


def _fixture_server(tmp_path: Path, bodies: dict[str, bytes]) -> tuple[str, list[dict]]:
    """Write fixture images under a ``file://`` root and their manifest rows.

    Parameters
    ----------
    tmp_path:
        Test scratch directory.
    bodies:
        File name to image bytes.

    Returns
    -------
    tuple[str, list[dict]]
        The ``base_url`` replacing the COCO prefix, and pinned rows.
    """

    root = tmp_path / "server"
    root.mkdir()
    rows = []
    for name, body in bodies.items():
        (root / name).write_bytes(body)
        rows.append(
            {
                "bytes": len(body),
                "file": name,
                "sha256": hashlib.sha256(body).hexdigest(),
                "url": COCO_URL_PREFIX + name,
            }
        )
    return root.as_uri() + "/", rows


def test_coco_fetch_verifies_and_caches_from_a_local_fixture(tmp_path: Path) -> None:
    """On-demand fetch writes verified bytes in manifest order and reuses the cache."""

    base_url, rows = _fixture_server(tmp_path, {"a.jpg": b"alpha" * 10, "b.jpg": b"beta" * 7})
    dest = tmp_path / "cache"

    paths = fetch_coco(dest, rows=rows, base_url=base_url)

    assert [path.name for path in paths] == ["a.jpg", "b.jpg"]
    assert paths[0].read_bytes() == b"alpha" * 10
    assert not list(dest.glob("*.part"))
    (tmp_path / "server" / "a.jpg").unlink()
    assert fetch_coco(dest, rows=rows, base_url=base_url) == paths


def test_coco_fetch_refuses_a_checksum_mismatch_typed(tmp_path: Path) -> None:
    """A changed source body refuses with CocoChecksumMismatch and keeps no file."""

    base_url, rows = _fixture_server(tmp_path, {"a.jpg": b"alpha" * 10})
    (tmp_path / "server" / "a.jpg").write_bytes(b"ALPHA" * 10)
    dest = tmp_path / "cache"

    with pytest.raises(CocoChecksumMismatch) as excinfo:
        fetch_coco(dest, rows=rows, base_url=base_url)

    assert excinfo.value.file == "a.jpg"
    assert excinfo.value.expected_sha256 == rows[0]["sha256"]
    assert excinfo.value.actual_sha256 == hashlib.sha256(b"ALPHA" * 10).hexdigest()
    assert not list(dest.iterdir())


def test_coco_fetch_refuses_a_size_mismatch_and_refetches_a_corrupt_cache(
    tmp_path: Path,
) -> None:
    """An oversized body refuses unhashed; a corrupted cached copy is fetched again."""

    base_url, rows = _fixture_server(tmp_path, {"a.jpg": b"alpha" * 10})
    dest = tmp_path / "cache"
    dest.mkdir()
    (dest / "a.jpg").write_bytes(b"stale")
    assert fetch_coco(dest, rows=rows, base_url=base_url)[0].read_bytes() == b"alpha" * 10

    (tmp_path / "server" / "a.jpg").write_bytes(b"alpha" * 11)
    (dest / "a.jpg").write_bytes(b"stale")
    with pytest.raises(CocoChecksumMismatch) as excinfo:
        fetch_coco(dest, rows=rows, base_url=base_url)
    assert excinfo.value.actual_sha256 is None
    assert excinfo.value.actual_bytes == rows[0]["bytes"] + 1


def test_coco_fetch_refuses_unpinned_urls_and_the_vendored_dir(tmp_path: Path) -> None:
    """Only pinned COCO URLs are fetched, and never into the vendored images."""

    row = dict(coco_manifest_rows()[0], url="http://example.invalid/x.jpg")
    with pytest.raises(ValueError, match="unpinned URL"):
        fetch_coco(tmp_path, rows=[row])
    with pytest.raises(ValueError, match="vendored images"):
        fetch_coco(IMAGES_DIR, rows=[])


@pytest.mark.rare
def test_coco_precision_corpus_real_fetch_matches_the_manifest(tmp_path: Path) -> None:
    """Opt-in (``-m rare``): fetch all 128 pinned COCO images and verify each."""

    paths = fetch_coco(tmp_path / "coco", timeout=120.0)

    assert len(paths) == 128
    rows = coco_manifest_rows()
    for path, row in zip(paths, rows, strict=True):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
