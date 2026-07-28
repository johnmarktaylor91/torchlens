"""Controlled fetch and literal evidence tests for crawler Slice D."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.evidence import EvidenceValidationError, validate_evidence
from menagerie.crawler.fetcher import (
    FetchHashMismatchError,
    FetchTarget,
    UnpinnedTargetError,
    cas_path,
    fetch_target,
)
from menagerie.crawler.identity import hash_bytes


def _evidence(text: str, digest: str) -> dict[str, Any]:
    """Build one complete literal-evidence block.

    Parameters
    ----------
    text:
        Verbatim source excerpt.
    digest:
        Hash of the exact excerpt bytes.

    Returns
    -------
    dict[str, Any]
        Evidence block.
    """

    return {
        "excerpts": [
            {
                "evidence_id": "e1",
                "source_id": "s1",
                "locator": f"bytes:0-{len(text.encode('utf-8'))}",
                "text": text,
                "text_sha256": digest,
                "supports": ["description"],
                "family_level": True,
                "disposition": "supporting",
                "license_disposition": "short-excerpt-committed",
            }
        ],
        "coverage": {
            "all_agent_fields_have_support": True,
            "missing_support": [],
            "family_grounding_complete": True,
        },
        "evidence_identity": digest,
        "family_grounding_path": None,
    }


def test_fetcher_stores_by_hash_and_refetch_is_idempotent(tmp_path: Path) -> None:
    """A verified CAS object is reused without calling the retriever again."""

    content = b"exact pinned source"
    digest = hash_bytes(content)
    target = FetchTarget("s1", "test://source/model", "commit-1", digest)
    calls = 0

    def retrieve(url: str) -> bytes:
        """Return fixed test bytes and count retrievals.

        Parameters
        ----------
        url:
            Exact target URL.

        Returns
        -------
        bytes
            Fixed source bytes.
        """

        nonlocal calls
        assert url == target.url
        calls += 1
        return content

    first = fetch_target(target, tmp_path, fetch_bytes=retrieve)
    second = fetch_target(target, tmp_path, fetch_bytes=retrieve)
    assert cas_path(tmp_path, digest).read_bytes() == content
    assert first["retrieval_status"] == "fetched"
    assert second["retrieval_status"] == "already-present"
    assert calls == 1


def test_fetcher_rejects_unpinned_and_mismatched_without_cas_write(tmp_path: Path) -> None:
    """Invalid pins and bytes fail before publishing a CAS object."""

    content = b"unexpected"
    expected = hash_bytes(b"expected")
    with pytest.raises(UnpinnedTargetError):
        fetch_target(
            FetchTarget("s1", "test://source/model", "", expected),
            tmp_path,
            fetch_bytes=lambda _url: content,
        )
    with pytest.raises(FetchHashMismatchError):
        fetch_target(
            FetchTarget("s1", "test://source/model", "commit-1", expected),
            tmp_path,
            fetch_bytes=lambda _url: content,
        )
    assert not cas_path(tmp_path, expected).exists()


def test_fetcher_accepts_an_absent_digest_and_pins_what_it_retrieved(tmp_path: Path) -> None:
    """The author cannot digest bytes it never fetched, so the fetch learns the pin."""

    content = b"a source the author named but never read"
    digest = hash_bytes(content)
    target = FetchTarget("s1", "test://source/model", "commit-1")
    assert target.expected_sha256 == ""

    manifest = fetch_target(target, tmp_path, fetch_bytes=lambda _url: content)

    assert manifest["content_sha256"] == digest
    assert manifest["retrieval_status"] == "fetched"
    assert manifest["fetched_bytes_len"] == len(content)
    assert Path(str(manifest["cas_path"])) == cas_path(tmp_path, digest)
    assert cas_path(tmp_path, digest).read_bytes() == content


def test_fetcher_accepts_the_bare_hex_digest_the_source_brief_advertises(tmp_path: Path) -> None:
    """`<64 hex>` and `sha256:<64 hex>` are the same pin and both are enforced."""

    content = b"exact pinned source"
    digest = hash_bytes(content)
    bare = digest.removeprefix("sha256:")

    manifest = fetch_target(
        FetchTarget("s1", "test://source/model", "commit-1", bare),
        tmp_path,
        fetch_bytes=lambda _url: content,
    )
    assert manifest["content_sha256"] == digest

    upper = fetch_target(
        FetchTarget("s1", "test://source/model", "commit-1", bare.upper()),
        tmp_path,
        fetch_bytes=lambda _url: content,
    )
    assert upper["content_sha256"] == digest


def test_fetcher_still_fails_loudly_on_a_wrong_supplied_digest(tmp_path: Path) -> None:
    """A digest the author did supply is enforced exactly, in every spelling."""

    content = b"the bytes actually served"
    wrong = hash_bytes(b"the bytes the author claimed")
    for declared in (wrong, wrong.removeprefix("sha256:")):
        with pytest.raises(FetchHashMismatchError):
            fetch_target(
                FetchTarget("s1", "test://source/model", "commit-1", declared),
                tmp_path,
                fetch_bytes=lambda _url: content,
            )
    assert not cas_path(tmp_path, wrong).exists()
    assert not cas_path(tmp_path, hash_bytes(content)).exists()


def test_fetcher_rejects_a_malformed_digest_instead_of_ignoring_it(tmp_path: Path) -> None:
    """A garbled pin is a contract defect, never silently downgraded to absence."""

    for declared in ("sha256:not-hex", "abc123", hash_bytes(b"x") + "0", "sha256:"):
        with pytest.raises(UnpinnedTargetError, match="expected_sha256"):
            fetch_target(
                FetchTarget("s1", "test://source/model", "commit-1", declared),
                tmp_path,
                fetch_bytes=lambda _url: b"content",
            )


def test_evidence_verbatim_locator_and_support_coverage(tmp_path: Path) -> None:
    """A matching literal excerpt at a fetched locator grounds its claim."""

    text = "A source-grounded architecture."
    content_hash = hash_bytes(text.encode())
    path = cas_path(tmp_path, content_hash)
    path.parent.mkdir(parents=True)
    path.write_bytes(text.encode())
    manifest = {
        "sources": [{"source_id": "s1", "content_sha256": content_hash, "cas_path": str(path)}]
    }
    report = validate_evidence(
        _evidence(text, hash_bytes(text.encode())),
        manifest,
        ["description"],
        require_family_grounding=True,
    )
    assert report.supported_claims == frozenset({"description"})


@pytest.mark.parametrize("failure", ["altered", "missing-support", "ungrounded"])
def test_evidence_rejects_altered_or_ungrounded_claims(tmp_path: Path, failure: str) -> None:
    """Hash mismatch, empty supports, and missing claim coverage all fail.

    Parameters
    ----------
    failure:
        Evidence corruption to apply.
    """

    text = "Literal source sentence."
    content_hash = hash_bytes(text.encode())
    path = cas_path(tmp_path, content_hash)
    path.parent.mkdir(parents=True)
    path.write_bytes(text.encode())
    manifest = {
        "sources": [{"source_id": "s1", "content_sha256": content_hash, "cas_path": str(path)}]
    }
    evidence = deepcopy(_evidence(text, hash_bytes(text.encode())))
    required = ["description"]
    if failure == "altered":
        evidence["excerpts"][0]["text"] = "Altered source sentence."
    elif failure == "missing-support":
        evidence["excerpts"][0]["supports"] = []
    else:
        required.append("citation")
    with pytest.raises(EvidenceValidationError):
        validate_evidence(evidence, manifest, required)
