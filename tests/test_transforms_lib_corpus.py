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

from tests.transforms_corpus.loader import (
    COCO_URL_PREFIX,
    coco_manifest_rows,
    fetch_coco,
)

pytest.importorskip("PIL", reason="the corpus loader decodes JPEG via PIL")


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


@pytest.mark.rare
def test_coco_precision_corpus_real_fetch_matches_the_manifest(tmp_path: Path) -> None:
    """Opt-in (``-m rare``): fetch all 128 pinned COCO images and verify each."""

    paths = fetch_coco(tmp_path / "coco", timeout=120.0)

    assert len(paths) == 128
    rows = coco_manifest_rows()
    for path, row in zip(paths, rows, strict=True):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
