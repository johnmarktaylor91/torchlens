"""Broker-to-CAS promotion, and the split between corruption and upstream drift.

The engine used to verify bytes it already possessed by fetching them a second
time from the live web. Pages carrying an embedded nonce, timestamp, session id,
or ad token differ on every retrieval, so that second fetch manufactured digest
disagreements out of nothing -- and the single exception type covering both that
and genuine local CAS corruption meant nothing could tell them apart without
parsing a message string.

Every fixture here uses MORE THAN ONE source on purpose: a single-source fixture
freezes the width of the thing under test and hides exactly the bug where one
row's bytes are attributed to another.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from menagerie.crawler.fetcher import (
    CasObjectCorruptError,
    DigestOrigin,
    FetchHashMismatchError,
    FetchTarget,
    UnpinnedTargetError,
    UpstreamContentDriftError,
    cas_path,
    fetch_target,
    fetch_targets,
)
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.source_broker import (
    SourceBrokerError,
    TransportResponse,
    broker_evidence_dirs,
    broker_source_pack,
    promote_broker_evidence,
)

RESOLVED_SHA = "8379e338134bd33e53340b47f95c13028a4f9dbf"
COMMITS_URL = "https://api.github.com/repos/pykeen/pykeen/commits/v1.11.1"
RAW_URL = (
    "https://raw.githubusercontent.com/pykeen/pykeen/"
    f"{RESOLVED_SHA}/src/pykeen/models/unimodal/mure.py"
)
PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC4522567/"

#: The live PMC page from the m3671 incident: two retrievals minutes apart
#: returned different bytes at IDENTICAL length, the signature of a fixed-width
#: embedded nonce or timestamp rather than of a substituted document.
PAGE_FIRST = b"<html><body>nonce=0001 the paper body that never changes</body></html>"
PAGE_DRIFTED = b"<html><body>nonce=9999 the paper body that never changes</body></html>"
IMPL_BODY = b"class MuRE:\n    pass\n"


class MapTransport:
    """In-memory transport: url -> (status, body)."""

    def __init__(self, responses: dict) -> None:
        self.responses = responses
        self.requested: list[str] = []

    def __call__(self, url: str, *, max_bytes: int, timeout: float) -> TransportResponse:
        """Serve one mapped response and record that the url was requested."""

        self.requested.append(url)
        status, body = self.responses.get(url, (0, b""))
        return TransportResponse(
            status=status,
            final_url=url,
            redirect_chain=(url,),
            body=body[:max_bytes],
            truncated=len(body) > max_bytes,
            error=None if status else "no route",
        )


def _descriptors() -> list[dict]:
    """Return two genuinely different sources: a forge file and a live page."""

    return [
        {
            "source_id": "impl-mure",
            "kind": "forge-file",
            "repo": "github.com/pykeen/pykeen",
            "path": "src/pykeen/models/unimodal/mure.py",
            "ref": "v1.11.1",
            "requested_role": "implementation",
            "media_type_hint": "text/x-python",
            "basis": "The upstream repository owns the MuRE implementation.",
        },
        {
            "source_id": "doc-paper",
            "kind": "raw-url",
            "url": PAPER_URL,
            "requested_role": "documentation",
            "media_type_hint": "text/html",
            "basis": "The introducing paper describes the architecture.",
        },
    ]


def _brokered(root: Path, page: bytes = PAGE_FIRST) -> tuple[list[dict], MapTransport]:
    """Run the REAL broker over both sources and return its rows."""

    transport = MapTransport(
        {
            COMMITS_URL: (200, json.dumps({"sha": RESOLVED_SHA}).encode()),
            RAW_URL: (200, IMPL_BODY),
            PAPER_URL: (200, page),
        }
    )
    pack = broker_source_pack(
        _descriptors(), broker_dir=root / "broker", transport=transport
    )
    assert len(pack.rows) == 2, "fixture must broker BOTH sources, not one"
    return pack.rows, transport


def _targets(rows: list[dict], *, lengths: Optional[dict] = None) -> list[FetchTarget]:
    """Build controlled-fetch targets exactly as the driver's author lane does."""

    lengths = lengths or {}
    return [
        FetchTarget(
            source_id=str(row["source_id"]),
            url=str(row["url"]),
            revision=str(row["revision"]),
            expected_sha256=str(row["expected_sha256"]),
            media_type=str(row["media_type"]),
            digest_origin=DigestOrigin.CONTROLLED_FETCH,
            expected_bytes_len=lengths.get(str(row["source_id"])),
        )
        for row in rows
    ]


@pytest.mark.smoke
def test_broker_bytes_promote_into_the_cas_so_no_second_fetch_ever_happens(
    tmp_path: Path,
) -> None:
    """The drift window is closed by not going back to the network at all."""

    root = tmp_path / "author"
    rows, transport = _brokered(root)
    brokered_urls = list(transport.requested)

    promotions = promote_broker_evidence(
        rows, broker_evidence_dirs(root), root / "source-cas"
    )

    assert [entry["disposition"] for entry in promotions] == ["promoted", "promoted"]
    assert {entry["source_id"] for entry in promotions} == {"impl-mure", "doc-paper"}
    # The two sources must promote to their OWN distinct bytes. A promotion that
    # cross-wired the rows would still report two successes.
    by_id = {str(entry["source_id"]): entry for entry in promotions}
    assert by_id["impl-mure"]["content_sha256"] == hash_bytes(IMPL_BODY)
    assert by_id["doc-paper"]["content_sha256"] == hash_bytes(PAGE_FIRST)
    assert by_id["impl-mure"]["bytes_len"] == len(IMPL_BODY)
    assert by_id["doc-paper"]["bytes_len"] == len(PAGE_FIRST)
    assert cas_path(root / "source-cas", hash_bytes(IMPL_BODY)).read_bytes() == IMPL_BODY
    assert cas_path(root / "source-cas", hash_bytes(PAGE_FIRST)).read_bytes() == PAGE_FIRST

    def refuse(url: str) -> bytes:
        """Fail the test if the controlled fetch reaches the network at all."""

        raise AssertionError(f"controlled fetch re-fetched {url!r} despite holding the bytes")

    manifest = fetch_targets(_targets(rows), root / "source-cas", fetch_bytes=refuse)

    sources = manifest["sources"]
    assert isinstance(sources, list) and len(sources) == 2
    assert [row["retrieval_status"] for row in sources] == [
        "already-present",
        "already-present",
    ]
    # No further network traffic beyond the broker's own three requests.
    assert transport.requested == brokered_urls


@pytest.mark.smoke
def test_a_drifting_upstream_document_produces_the_typed_drift_outcome(
    tmp_path: Path,
) -> None:
    """Both digests and both byte lengths are recorded, and the lane survives."""

    root = tmp_path / "author"
    rows, _ = _brokered(root)
    lengths = {"impl-mure": len(IMPL_BODY), "doc-paper": len(PAGE_FIRST)}
    # Promotion is deliberately skipped so the fetch is forced back to a network
    # that now serves different bytes for the paper -- the exact m3671 shape.
    served = {RAW_URL: IMPL_BODY, PAPER_URL: PAGE_DRIFTED}

    paper = next(target for target in _targets(rows, lengths=lengths) if target.source_id == "doc-paper")
    with pytest.raises(UpstreamContentDriftError) as caught:
        fetch_target(paper, root / "source-cas", fetch_bytes=lambda url: served[url])

    drift = caught.value
    assert drift.expected_sha256 == hash_bytes(PAGE_FIRST)
    assert drift.actual_sha256 == hash_bytes(PAGE_DRIFTED)
    # The two slots must genuinely differ. An assertion that passes because both
    # sides hold the same placeholder proves nothing at all.
    assert drift.expected_sha256 != drift.actual_sha256
    assert drift.expected_bytes_len == len(PAGE_FIRST)
    assert drift.actual_bytes_len == len(PAGE_DRIFTED)
    assert drift.expected_bytes_len == drift.actual_bytes_len, "the incident's fixed-width nonce"
    assert drift.url == PAPER_URL
    assert str(drift) == (
        f"upstream content drift for {PAPER_URL!r}: expected {hash_bytes(PAGE_FIRST)} "
        f"({len(PAGE_FIRST)} bytes), got {hash_bytes(PAGE_DRIFTED)} "
        f"({len(PAGE_DRIFTED)} bytes)"
    )

    # The plural entry point resolves it into a RECORDED disposition instead of
    # killing the lane, and re-pins the row to the bytes upstream now serves so
    # every excerpt locator is re-verified against those bytes.
    manifest = fetch_targets(
        _targets(rows, lengths=lengths),
        root / "source-cas",
        fetch_bytes=lambda url: served[url],
    )
    sources = manifest["sources"]
    assert isinstance(sources, list)
    by_id = {str(row["source_id"]): row for row in sources}

    assert "upstream_drift" not in by_id["impl-mure"], "a stable source must not be flagged"
    assert by_id["impl-mure"]["content_sha256"] == hash_bytes(IMPL_BODY)

    drifted = by_id["doc-paper"]
    assert drifted["content_sha256"] == hash_bytes(PAGE_DRIFTED)
    assert drifted["fetched_bytes_len"] == len(PAGE_DRIFTED)
    # Still admitted for excerpt verification: noticing drift must never be a
    # back door to silently skipping the excerpt check.
    assert drifted["retrieval_status"] == "fetched"
    assert drifted["upstream_drift"] == {
        "event": "upstream-content-drift",
        "source_id": "doc-paper",
        "url": PAPER_URL,
        "captured_sha256": hash_bytes(PAGE_FIRST),
        "captured_bytes_len": len(PAGE_FIRST),
        "refetched_sha256": hash_bytes(PAGE_DRIFTED),
        "refetched_bytes_len": len(PAGE_DRIFTED),
        "equal_length": True,
    }
    # The re-pinned bytes are the ones actually on disk, so the locator is
    # re-verified against the NEW document rather than the vanished one.
    assert Path(str(drifted["cas_path"])).read_bytes() == PAGE_DRIFTED


@pytest.mark.smoke
def test_a_corrupt_cas_object_is_still_fatal_and_is_not_a_drift(tmp_path: Path) -> None:
    """Our own store failing its own self-check can never be softened."""

    root = tmp_path / "author"
    rows, _ = _brokered(root)
    cas_root = root / "source-cas"
    promote_broker_evidence(rows, broker_evidence_dirs(root), cas_root)

    # Damage the object in place: right address, wrong bytes.
    damaged = cas_path(cas_root, hash_bytes(PAGE_FIRST))
    damaged.write_bytes(b"a truncated or scribbled-over object")

    paper = next(target for target in _targets(rows) if target.source_id == "doc-paper")
    with pytest.raises(CasObjectCorruptError) as caught:
        fetch_target(paper, cas_root, fetch_bytes=lambda _url: PAGE_FIRST)

    corrupt = caught.value
    assert corrupt.expected_sha256 == hash_bytes(PAGE_FIRST)
    assert corrupt.actual_sha256 == hash_bytes(b"a truncated or scribbled-over object")
    assert corrupt.expected_sha256 != corrupt.actual_sha256
    assert str(corrupt) == (
        f"corrupt CAS object {damaged}: expected {corrupt.expected_sha256}, "
        f"got {corrupt.actual_sha256}"
    )
    # Fatal means fatal: it is NOT the drift class, so nothing that resolves
    # drift can ever resolve this.
    assert not isinstance(corrupt, UpstreamContentDriftError)

    # And the plural entry point refuses it too, rather than re-pinning it.
    with pytest.raises(CasObjectCorruptError):
        fetch_targets(_targets(rows), cas_root, fetch_bytes=lambda _url: PAGE_FIRST)


@pytest.mark.smoke
def test_an_author_supplied_digest_unbacked_by_our_fetch_is_still_hard_refused(
    tmp_path: Path,
) -> None:
    """The anti-substitution tripwire is untouched where it actually applies."""

    cas_root = tmp_path / "source-cas"
    served = b"the bytes the server actually returned"
    claimed = hash_bytes(b"the bytes the author claimed were there")

    # Default origin is DECLARED: a digest nothing of ours ever fetched.
    declared = FetchTarget("s1", "test://source/model", "commit-1", claimed)
    assert declared.digest_origin is DigestOrigin.DECLARED

    with pytest.raises(FetchHashMismatchError) as caught:
        fetch_target(declared, cas_root, fetch_bytes=lambda _url: served)

    refusal = caught.value
    assert str(refusal) == (
        f"hash mismatch for 'test://source/model': expected {claimed}, "
        f"got {hash_bytes(served)}"
    )
    # It must NOT be reclassified as drift, which is the softer arm.
    assert not isinstance(refusal, UpstreamContentDriftError)
    assert not isinstance(refusal, CasObjectCorruptError)

    # The plural entry point resolves ONLY drift. A declared-digest violation
    # still propagates and still publishes nothing.
    with pytest.raises(FetchHashMismatchError):
        fetch_targets(
            [declared, FetchTarget("s2", "test://source/other", "commit-1")],
            cas_root,
            fetch_bytes=lambda _url: served,
        )
    assert not cas_path(cas_root, claimed).exists()
    assert not cas_path(cas_root, hash_bytes(served)).exists()


def test_promotion_refuses_a_blob_that_does_not_hash_to_its_own_row(
    tmp_path: Path,
) -> None:
    """The abbreviated evidence filename is a hint, never an authority."""

    root = tmp_path / "author"
    rows, _ = _brokered(root)
    cas_root = root / "source-cas"

    # Overwrite one evidence blob, keeping its 16-hex name, with bytes that do
    # not hash to the digest the row pins. This is the shape a prefix collision
    # or a tampered store would take.
    digest = hash_bytes(PAGE_FIRST)
    blob = root / "broker" / "evidence" / f"{digest.removeprefix('sha256:')[:16]}.bin"
    assert blob.is_file(), "fixture must actually target a real evidence blob"
    blob.write_bytes(b"substituted bytes wearing the right filename")

    promotions = promote_broker_evidence(rows, broker_evidence_dirs(root), cas_root)

    by_id = {str(entry["source_id"]): entry for entry in promotions}
    assert by_id["doc-paper"]["disposition"] == "evidence-digest-mismatch"
    # The untouched sibling still promotes, proving the refusal is per-source
    # and not a blanket bail-out.
    assert by_id["impl-mure"]["disposition"] == "promoted"
    assert not cas_path(cas_root, digest).exists()
    assert cas_path(cas_root, hash_bytes(IMPL_BODY)).read_bytes() == IMPL_BODY


def test_the_broker_still_refuses_an_author_supplied_digest_outright(
    tmp_path: Path,
) -> None:
    """No descriptor may carry a digest, so no lane digest is ever author-owned."""

    descriptor = {**_descriptors()[1], "expected_sha256": hash_bytes(b"anything")}
    with pytest.raises(SourceBrokerError) as caught:
        broker_source_pack([descriptor], broker_dir=tmp_path / "broker", transport=MapTransport({}))
    assert "machine-owned field 'expected_sha256'" in str(caught.value)


def test_a_malformed_digest_is_still_a_contract_defect_under_every_origin(
    tmp_path: Path,
) -> None:
    """Opting into controlled-fetch provenance never relaxes pin validation."""

    for origin in (DigestOrigin.DECLARED, DigestOrigin.CONTROLLED_FETCH):
        with pytest.raises(UnpinnedTargetError, match="expected_sha256"):
            fetch_target(
                FetchTarget(
                    "s1",
                    "test://source/model",
                    "commit-1",
                    "sha256:not-hex",
                    digest_origin=origin,
                ),
                tmp_path,
                fetch_bytes=lambda _url: b"whatever",
            )


def test_the_evidence_search_covers_every_producer_including_the_supplement(
    tmp_path: Path,
) -> None:
    """Three producers write evidence under one author root, not one.

    The driver's own stage-1 pass writes ``broker/evidence``; an operator that
    brokered its own pack writes under ``attempts/<id>/broker/``; and the single
    supplementary round writes ``broker/supplement/evidence``. A search that
    only knew about the first would leave the other two re-fetching from the
    network, which is the same seam this change exists to close.
    """

    root = tmp_path / "author"
    for relative in (
        "broker/evidence",
        "attempts/a1/broker/evidence",
        "attempts/a1/broker/supplement/evidence",
    ):
        (root / relative).mkdir(parents=True)
    (root / "attempts" / "a1" / "scratch").mkdir(parents=True)

    found = [str(path.relative_to(root)) for path in broker_evidence_dirs(root)]

    assert found == [
        "broker/evidence",
        "attempts/a1/broker/evidence",
        "attempts/a1/broker/supplement/evidence",
    ]
