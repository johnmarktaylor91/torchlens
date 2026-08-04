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


#: The exact shape that burned model ``m8245``. ar5iv renders LaTeX inter-author spacing
#: as U+2003 EM SPACE; an author reading the rendered page transcribes the ordinary space
#: it renders as, and every other character -- including the generated ``id`` attribute
#: values -- comes back exactly right.
_AR5IV_PAGE = (
    'prelude <br class="ltx_break"><sup id="id9.9.id9" class="ltx_sup">1</sup>Sea AI Lab\n'
    ' <sup id="id10.10.id10" class="ltx_sup">2</sup>National University of Singapore\n tail'
)
_TRANSCRIBED = (
    '<br class="ltx_break"><sup id="id9.9.id9" class="ltx_sup">1</sup>Sea AI Lab\n'
    ' <sup id="id10.10.id10" class="ltx_sup">2</sup>National University of Singapore'
)


def _space_case(tmp_path: Path, excerpt_text: str, *, byte_locator: bool) -> Any:
    """Validate one excerpt against the ar5iv page fixture."""

    content_hash = hash_bytes(_AR5IV_PAGE.encode())
    path = cas_path(tmp_path, content_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_AR5IV_PAGE.encode())
    manifest = {
        "sources": [{"source_id": "s1", "content_sha256": content_hash, "cas_path": str(path)}]
    }
    evidence = deepcopy(_evidence(excerpt_text, hash_bytes(excerpt_text.encode())))
    if not byte_locator:
        evidence["excerpts"][0]["locator"] = "author affiliation block"
    else:
        start = len("prelude ".encode())
        evidence["excerpts"][0]["locator"] = f"bytes:{start}-{start + len(_TRANSCRIBED.encode()) + 2}"
    return validate_evidence(evidence, manifest, ["description"], require_family_grounding=True)


def test_excerpt_differing_only_in_which_space_character_is_verbatim(tmp_path: Path) -> None:
    """A U+2003 the author typed as U+0020 is the same text, not a different one."""

    assert _TRANSCRIBED not in _AR5IV_PAGE  # raw bytes genuinely differ
    report = _space_case(tmp_path, _TRANSCRIBED, byte_locator=False)
    assert report.supported_claims == frozenset({"description"})


def test_whitespace_run_and_newline_differences_are_verbatim(tmp_path: Path) -> None:
    """Whitespace runs collapse, so a newline or a longer gap reads the same."""

    variant = _TRANSCRIBED.replace("Lab\n ", "Lab \n\n   ")
    assert _space_case(tmp_path, variant, byte_locator=False).excerpt_count == 1


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(("Sea AI Lab", "Sae AI Lab"), id="visible-character-changed"),
        pytest.param(("Sea AI Lab", "Sea AI Research Lab"), id="word-inserted"),
        pytest.param(("National University of", "University of"), id="word-deleted"),
        pytest.param(("Sea AI Lab", "SeaAILab"), id="space-deleted-tokens-joined"),
        pytest.param(("Singapore", "Singa pore"), id="space-inserted-token-split"),
        pytest.param(("id10.10.id10", "id10.11.id10"), id="generated-id-digit-changed"),
        pytest.param(("Sea AI Lab", "sea ai lab"), id="case-changed"),
        pytest.param(("Sea AI Lab", "Sea AI​ Lab"), id="zero-width-space-inserted"),
    ],
)
def test_non_whitespace_differences_are_still_refused(
    tmp_path: Path, mutation: tuple[str, str]
) -> None:
    """The fold erases only which space character was typed.

    Every visible character, its order, and every token boundary must still match, so a
    fabricated excerpt is refused exactly as before. U+200B is a format character rather
    than a separator and is deliberately not folded: it occupies no rendered gap, so
    treating it as whitespace would let an excerpt differ where nothing is displayed.
    """

    with pytest.raises(EvidenceValidationError, match="not verbatim in its fetched source"):
        _space_case(tmp_path, _TRANSCRIBED.replace(*mutation), byte_locator=False)


def test_byte_locator_branch_shares_the_same_space_equivalence(tmp_path: Path) -> None:
    """An author using the more precise locator is not punished harder for a space.

    Keeping the byte-range branch stricter than the membership branch would push
    authors toward the vaguer locator, which is the opposite of what it exists for.
    """

    assert _space_case(tmp_path, _TRANSCRIBED, byte_locator=True).excerpt_count == 1


def test_byte_locator_still_refuses_an_altered_range(tmp_path: Path) -> None:
    """The range anchor survives: text that is not at the locator still fails."""

    with pytest.raises(EvidenceValidationError, match="does not exist verbatim at"):
        _space_case(tmp_path, _TRANSCRIBED.replace("Sea AI Lab", "Sae AI Lab"), byte_locator=True)
