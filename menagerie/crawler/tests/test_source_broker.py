"""Source broker: machine-derived exact strings, typed per-target outcomes."""

from __future__ import annotations

import email
import json
import urllib.error
from pathlib import Path
from typing import Optional

import pytest

from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.source_broker import (
    OUTCOME_BAD_REF,
    OUTCOME_FETCHED,
    OUTCOME_OVERSIZED,
    OUTCOME_PAPER_DERIVATION_ONLY,
    OUTCOME_PROBED,
    OUTCOME_REDIRECT_REFUSED,
    OUTCOME_UNFETCHABLE_BY_POLICY,
    OUTCOME_UNREACHABLE,
    ROLE_DOCUMENTATION,
    ROLE_IMPLEMENTATION,
    ROLE_INTRODUCING_PAPER,
    FixtureTransport,
    RedirectRefused,
    SourceBrokerError,
    TransportResponse,
    UrllibTransport,
    _derive_media_type,
    broker_source_pack,
    derive_paper_metadata,
    write_broker_outputs,
)

RESOLVED_SHA = "8379e338134bd33e53340b47f95c13028a4f9dbf"
COMMITS_URL = "https://api.github.com/repos/pykeen/pykeen/commits/v1.11.1"
RAW_URL = (
    "https://raw.githubusercontent.com/pykeen/pykeen/"
    f"{RESOLVED_SHA}/src/pykeen/models/unimodal/mure.py"
)
ARXIV_URL = "https://export.arxiv.org/api/query?id_list=1905.09791"

ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Multi-relational Poincaré Graph Embeddings</title>
    <author><name>Ivana Balažević</name></author>
    <author><name>Carl Allen</name></author>
    <published>2019-05-23T17:59:59Z</published>
  </entry>
</feed>
"""


class MapTransport:
    """In-memory transport: url -> (status, body) or a raised refusal."""

    def __init__(self, responses: dict, refuse: Optional[dict] = None) -> None:
        self.responses = responses
        self.refuse = refuse or {}
        self.requested: list[str] = []

    def __call__(self, url: str, *, max_bytes: int, timeout: float) -> TransportResponse:
        self.requested.append(url)
        if url in self.refuse:
            raise RedirectRefused((url, self.refuse[url]), self.refuse[url])
        status, body = self.responses.get(url, (0, b""))
        truncated = len(body) > max_bytes
        return TransportResponse(
            status=status,
            final_url=url,
            redirect_chain=(url,),
            body=body[:max_bytes],
            truncated=truncated,
            error=None if status else "no route",
        )


def _impl_descriptor() -> dict:
    return {
        "source_id": "impl-mure",
        "kind": "forge-file",
        "repo": "github.com/pykeen/pykeen",
        "path": "src/pykeen/models/unimodal/mure.py",
        "ref": "v1.11.1",
        "requested_role": "implementation",
        "media_type_hint": "text/x-python",
        "basis": "The upstream repository owns the MuRE implementation.",
    }


def test_forge_ref_resolves_to_machine_derived_sha(tmp_path: Path) -> None:
    """The manifest revision is the resolver's SHA, never a model string."""

    body = b"class MuRE: pass\n"
    transport = MapTransport(
        {
            COMMITS_URL: (200, json.dumps({"sha": RESOLVED_SHA}).encode()),
            RAW_URL: (200, body),
        }
    )
    pack = broker_source_pack([_impl_descriptor()], broker_dir=tmp_path, transport=transport)
    assert len(pack.rows) == 1
    row = pack.rows[0]
    assert row["revision"] == RESOLVED_SHA
    assert row["url"] == RAW_URL
    assert row["expected_sha256"] == hash_bytes(body)
    outcome = pack.outcomes[0]
    assert outcome.outcome == OUTCOME_FETCHED
    receipt = outcome.resolver_receipt
    assert receipt is not None and receipt["resolved_sha"] == RESOLVED_SHA
    assert receipt["endpoint"] == COMMITS_URL
    assert receipt["response_sha256"] is not None


def test_descriptor_smuggling_exact_strings_is_rejected(tmp_path: Path) -> None:
    """A model-supplied SHA or digest never enters the pack."""

    descriptor = {**_impl_descriptor(), "expected_sha256": "sha256:" + "0" * 64}
    with pytest.raises(SourceBrokerError, match="machine-owned"):
        broker_source_pack([descriptor], broker_dir=tmp_path, transport=MapTransport({}))


@pytest.mark.parametrize("value", ["fabricated", "", None])
@pytest.mark.parametrize(
    "field",
    [
        "revision",
        "commit_sha",
        "expected_sha256",
        "content_sha256",
        "sha256",
        "final_url",
        "redirect_chain",
        "resolver_receipt",
        "broker_role",
        "broker_citable_role",
        "media_type",
        "derived_citation",
        "retrieved_at",
    ],
)
def test_descriptor_smuggling_is_rejected_on_presence(
    tmp_path: Path, field: str, value: object
) -> None:
    """A forbidden identity field is rejected even when empty or null."""

    descriptor = {**_impl_descriptor(), field: value}
    with pytest.raises(SourceBrokerError, match="machine-owned"):
        broker_source_pack([descriptor], broker_dir=tmp_path, transport=MapTransport({}))


def test_bad_ref_is_a_typed_outcome_with_receipt(tmp_path: Path) -> None:
    """An unresolvable ref yields ``bad-ref`` plus the forge receipt, no row."""

    transport = MapTransport({COMMITS_URL: (422, b'{"message":"No commit found"}')})
    pack = broker_source_pack([_impl_descriptor()], broker_dir=tmp_path, transport=transport)
    assert pack.rows == []
    outcome = pack.outcomes[0]
    assert outcome.outcome == OUTCOME_BAD_REF
    assert outcome.resolver_receipt is not None
    assert outcome.resolver_receipt["status"] == 422


def test_manifest_transport_and_classification_facts_are_broker_derived(
    tmp_path: Path,
) -> None:
    """Requested role/media hints and the pre-redirect URL cannot control a row."""

    requested_url = "https://example.org/readme.md"
    final_url = "https://raw.githubusercontent.com/acme/widgets/main/model.py"
    body = b"class Model:\n    pass\n"

    def transport(url: str, *, max_bytes: int, timeout: float) -> TransportResponse:
        """Return one redirected Python source response."""

        del max_bytes, timeout
        assert url == requested_url
        return TransportResponse(
            status=200,
            final_url=final_url,
            redirect_chain=(requested_url, final_url),
            body=body,
            truncated=False,
        )

    pack = broker_source_pack(
        [
            {
                "source_id": "spoof-attempt",
                "kind": "raw-url",
                "url": requested_url,
                "requested_role": "documentation",
                "media_type_hint": "application/x-authored-spoof",
                "basis": "The final object is a source file.",
            }
        ],
        broker_dir=tmp_path,
        transport=transport,
    )

    assert len(pack.rows) == 1
    row = pack.rows[0]
    assert row["url"] == final_url
    assert row["final_url"] == final_url
    assert row["broker_role"] == ROLE_IMPLEMENTATION
    assert row["media_type"] == "text/x-python"
    assert row["media_type_method"] == "path-extension"
    assert row["requested_url"] == requested_url
    assert row["requested_role"] == "documentation"
    assert row["media_type_hint"] == "application/x-authored-spoof"
    assert row["revision"] == hash_bytes(body)
    assert "role" not in row


def test_per_target_outcomes_are_independent(tmp_path: Path) -> None:
    """One dead target never aborts the pack: each yields its own outcome."""

    body = b"ok"
    transport = MapTransport(
        {
            COMMITS_URL: (200, json.dumps({"sha": RESOLVED_SHA}).encode()),
            RAW_URL: (200, body),
            "https://example.org/dead": (404, b""),
        },
        refuse={"https://example.org/hop": "https://evil.example/x"},
    )
    descriptors = [
        _impl_descriptor(),
        {
            "source_id": "doc-dead",
            "kind": "raw-url",
            "url": "https://example.org/dead",
            "requested_role": "documentation",
            "basis": "Candidate documentation endpoint.",
        },
        {
            "source_id": "doc-hop",
            "kind": "raw-url",
            "url": "https://example.org/hop",
            "requested_role": "documentation",
            "basis": "Candidate documentation endpoint.",
        },
    ]
    pack = broker_source_pack(descriptors, broker_dir=tmp_path, transport=transport)
    outcomes = {item.source_id: item.outcome for item in pack.outcomes}
    assert outcomes == {
        "impl-mure": OUTCOME_FETCHED,
        "doc-dead": OUTCOME_UNREACHABLE,
        "doc-hop": OUTCOME_REDIRECT_REFUSED,
    }
    assert [row["source_id"] for row in pack.rows] == ["impl-mure"]


def test_oversized_target_is_typed_and_carries_no_digest(tmp_path: Path) -> None:
    """A body over the ceiling becomes ``oversized`` with no manifest row."""

    transport = MapTransport({"https://example.org/big": (200, b"x" * 4096)})
    pack = broker_source_pack(
        [
            {
                "source_id": "doc-big",
                "kind": "raw-url",
                "url": "https://example.org/big",
                "requested_role": "documentation",
                "basis": "Candidate documentation endpoint.",
            }
        ],
        broker_dir=tmp_path,
        transport=transport,
        target_byte_ceiling=1024,
    )
    assert pack.rows == []
    assert pack.outcomes[0].outcome == OUTCOME_OVERSIZED
    assert pack.outcomes[0].sha256 is None


def test_probe_targets_yield_probe_receipts_not_rows(tmp_path: Path) -> None:
    """Negative-proof candidates are probed and receipted, never manifest rows."""

    transport = MapTransport({"https://example.org/candidate": (200, b"page")})
    pack = broker_source_pack(
        [
            {
                "source_id": "probe-1",
                "kind": "raw-url",
                "url": "https://example.org/candidate",
                "requested_role": "probe",
                "basis": "Candidate requested only for a negative-proof probe.",
            }
        ],
        broker_dir=tmp_path,
        transport=transport,
    )
    assert pack.rows == []
    assert pack.outcomes[0].outcome == OUTCOME_PROBED
    assert pack.outcomes[0].sha256 is not None


def test_paper_role_requires_derived_metadata(tmp_path: Path) -> None:
    """``introducing-paper`` binds only when the registry derivation succeeds."""

    transport = MapTransport({ARXIV_URL: (200, ARXIV_ATOM.encode("utf-8"))})
    pack = broker_source_pack(
        [
            {
                "source_id": "paper-mure",
                "kind": "paper",
                "url": "https://arxiv.org/abs/1905.09791",
                "requested_role": "paper",
                "basis": "Candidate introducing paper.",
            }
        ],
        broker_dir=tmp_path,
        transport=transport,
    )
    outcome = pack.outcomes[0]
    assert outcome.bound_role == ROLE_INTRODUCING_PAPER
    assert outcome.outcome == OUTCOME_PAPER_DERIVATION_ONLY
    citation = pack.derived_citations[0]
    assert citation["title"] == "Multi-relational Poincaré Graph Embeddings"
    assert citation["authors"] == ["Ivana Balažević", "Carl Allen"]
    assert citation["year"] == 2019
    assert citation["identifiers"] == {"arxiv": "1905.09791"}


def test_paper_derivation_failure_never_binds_the_paper_role(tmp_path: Path) -> None:
    """A dead registry means no ``introducing-paper`` role, typed unreachable."""

    pack = broker_source_pack(
        [
            {
                "source_id": "paper-x",
                "kind": "paper",
                "url": "https://arxiv.org/abs/1905.09791",
                "requested_role": "paper",
                "basis": "Candidate introducing paper.",
            }
        ],
        broker_dir=tmp_path,
        transport=MapTransport({}),
    )
    outcome = pack.outcomes[0]
    assert outcome.bound_role is None
    assert outcome.outcome == OUTCOME_UNREACHABLE
    assert pack.derived_citations == []


def test_crossref_derivation(tmp_path: Path) -> None:
    """DOIs derive through Crossref with the raw response digested."""

    doi = "10.5555/12345678"
    endpoint = f"https://api.crossref.org/works/{doi}"
    message = {
        "message": {
            "title": ["A Very Real Paper"],
            "author": [{"given": "Ada", "family": "Lovelace"}],
            "issued": {"date-parts": [[1843]]},
            "container-title": ["Journal of Engines"],
        }
    }
    transport = MapTransport({endpoint: (200, json.dumps(message).encode())})
    citation, receipt = derive_paper_metadata(
        f"https://doi.org/{doi}",
        transport=transport,
        evidence_dir=tmp_path / "evidence",
        clock=lambda: "2026-01-01T00:00:00Z",
    )
    assert citation is not None
    assert citation["title"] == "A Very Real Paper"
    assert citation["authors"] == ["Ada Lovelace"]
    assert citation["year"] == 1843
    assert citation["venue"] == "Journal of Engines"
    assert receipt["registry"] == "crossref"


def test_total_byte_ceiling_stops_later_fetches_typed(tmp_path: Path) -> None:
    """Exhausting the total budget yields typed outcomes, not silence."""

    transport = MapTransport(
        {
            "https://example.org/a": (200, b"a" * 100),
            "https://example.org/b": (200, b"b" * 100),
        }
    )
    pack = broker_source_pack(
        [
            {
                "source_id": "a",
                "kind": "raw-url",
                "url": "https://example.org/a",
                "requested_role": "documentation",
                "basis": "First documentation object.",
            },
            {
                "source_id": "b",
                "kind": "raw-url",
                "url": "https://example.org/b",
                "requested_role": "documentation",
                "basis": "Second documentation object.",
            },
        ],
        broker_dir=tmp_path,
        transport=transport,
        total_byte_ceiling=100,
    )
    assert pack.outcomes[0].outcome == OUTCOME_FETCHED
    assert pack.outcomes[1].outcome == OUTCOME_UNREACHABLE
    assert "ceiling" in pack.outcomes[1].detail


def test_write_broker_outputs_persists_receipts(tmp_path: Path) -> None:
    """Receipts persist with the pack version and full outcome table."""

    body = b"class MuRE: pass\n"
    transport = MapTransport(
        {
            COMMITS_URL: (200, json.dumps({"sha": RESOLVED_SHA}).encode()),
            RAW_URL: (200, body),
        }
    )
    pack = broker_source_pack([_impl_descriptor()], broker_dir=tmp_path, transport=transport)
    path = write_broker_outputs(pack, tmp_path)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["pack_version"].startswith("menagerie.crawler.source-broker-pack")
    assert persisted["broker"]["outcomes"][0]["resolver_receipt"]["resolved_sha"] == RESOLVED_SHA
    # Evidence bytes were stored content-addressed.
    evidence = list((tmp_path / "evidence").iterdir())
    assert evidence


# -- W-1: the sniff must record the truth for extensionless HTML ------------

#: Replay-shaped after the frozen rung-8 m5273 manifest: arXiv abs pages and
#: ar5iv renderings are extensionless and begin with a doctype, and the old
#: sniff recorded them ``text/plain`` -- a machine falsehood the byte-exact
#: staging echo then forced authors to repeat (an author writing the true
#: ``text/html`` died at staging AFTER passing result validation).
ARXIV_ABS_SHAPED = b'<!DOCTYPE html>\n<html lang="en">\n\n<head><script>x</script></head></html>'
AR5IV_SHAPED = b'<!DOCTYPE html><html lang="en">\n<head>\n<meta charset="utf-8"></head></html>'


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (ARXIV_ABS_SHAPED, "text/html"),
        (AR5IV_SHAPED, "text/html"),
        (b"  \r\n\t<HTML LANG='EN'><body></body></HTML>", "text/html"),
        (b"\xef\xbb\xbf<!doctype html><p>bom-prefixed page</p>", "text/html"),
        (b"<!-- comment first --><html></html>", "text/html"),
        (b"<div class='x'>fragment without doctype</div>", "text/html"),
        # Negatives: the HTML branch must not swallow the other truths.
        (b"plain prose about <angles> that opens no tag", "text/plain"),
        (b"<abbrev-tag> is not a recognized html opener", "text/plain"),
        (b'{"a": 1}', "application/json"),
        (b"\x89PNG\r\n\x1a\n", "application/octet-stream"),
    ],
)
def test_extensionless_body_sniffs_to_its_true_media_type(body: bytes, expected: str) -> None:
    """Extensionless HTML is recorded ``text/html``; non-HTML stays untouched."""

    media_type, method = _derive_media_type("https://arxiv.org/abs/2303.05499", body)
    assert media_type == expected
    assert method == "content-sniff"


def test_path_extension_still_beats_the_content_sniff() -> None:
    """A telling suffix remains authoritative over the body bytes."""

    media_type, method = _derive_media_type("https://x.example/model.py", ARXIV_ABS_SHAPED)
    assert media_type == "text/x-python"
    assert method == "path-extension"


# -- W-6: retrieved_at is machine-stamped, never authored --------------------


def test_manifest_rows_carry_machine_stamped_retrieved_at(tmp_path: Path) -> None:
    """Every manifest row carries the broker clock's own per-fetch instant."""

    ticks = iter(
        [
            "2026-08-05T05:10:43Z",  # forge ref-resolution receipt
            "2026-08-05T05:10:44Z",  # forge fetch completion
            "2026-08-05T05:10:45Z",  # raw-url fetch completion
        ]
    )
    body = b"class MuRE: pass\n"
    page = b"<!DOCTYPE html><html><head><title>paper</title></head></html>"
    transport = MapTransport(
        {
            COMMITS_URL: (200, json.dumps({"sha": RESOLVED_SHA}).encode()),
            RAW_URL: (200, body),
            "https://example.org/page": (200, page),
        }
    )
    pack = broker_source_pack(
        [
            _impl_descriptor(),
            {
                "source_id": "doc-page",
                "kind": "raw-url",
                "url": "https://example.org/page",
                "requested_role": "documentation",
                "basis": "Candidate documentation endpoint.",
            },
        ],
        broker_dir=tmp_path,
        transport=transport,
        clock=lambda: next(ticks),
    )
    stamped = {row["source_id"]: row["retrieved_at"] for row in pack.rows}
    assert stamped == {
        "impl-mure": "2026-08-05T05:10:44Z",
        "doc-page": "2026-08-05T05:10:45Z",
    }
    persisted = {item.source_id: item.to_dict()["retrieved_at"] for item in pack.outcomes}
    assert persisted == dict(stamped)


# -- W-10: the broker owns the citable paper role -----------------------------


def _paper_page_descriptor() -> dict:
    return {
        "source_id": "paper-abs",
        "kind": "raw-url",
        "url": "https://example.org/abs/2303.05499",
        "requested_role": "paper",
        "media_type_hint": "text/html",
        "basis": "The arXiv abstract page carries the citation facts.",
    }


def test_raw_url_paper_request_binds_the_citable_paper_role(tmp_path: Path) -> None:
    """A fetched document requested as the paper earns a machine-derived paper role.

    The manifest row keeps the lane's closed ``broker_role`` vocabulary
    (``documentation``) and records the broker's citation authority beside it,
    so an author's declared ``introducing-paper`` role finally has a
    machine-derived value to be checked against.
    """

    page = b"<!DOCTYPE html><html><head><title>GroundingDINO</title></head></html>"
    transport = MapTransport({"https://example.org/abs/2303.05499": (200, page)})
    pack = broker_source_pack([_paper_page_descriptor()], broker_dir=tmp_path, transport=transport)
    outcome = pack.outcomes[0]
    assert outcome.outcome == OUTCOME_FETCHED
    assert outcome.bound_role == ROLE_INTRODUCING_PAPER
    assert len(pack.rows) == 1
    row = pack.rows[0]
    assert row["broker_role"] == ROLE_DOCUMENTATION
    assert row["broker_citable_role"] == ROLE_INTRODUCING_PAPER
    assert row["media_type"] == "text/html"
    assert row["requested_role"] == "paper"


def test_paper_request_never_relabels_a_code_file(tmp_path: Path) -> None:
    """An implementation suffix beats the requested paper role."""

    transport = MapTransport({"https://example.org/model.py": (200, b"class Model: pass\n")})
    pack = broker_source_pack(
        [
            {
                "source_id": "spoof-paper",
                "kind": "raw-url",
                "url": "https://example.org/model.py",
                "requested_role": "paper",
                "basis": "Mislabeled code file.",
            }
        ],
        broker_dir=tmp_path,
        transport=transport,
    )
    outcome = pack.outcomes[0]
    assert outcome.bound_role == ROLE_IMPLEMENTATION
    row = pack.rows[0]
    assert row["broker_role"] == ROLE_IMPLEMENTATION
    assert "broker_citable_role" not in row


def test_paper_request_for_a_binary_blob_stays_documentation(tmp_path: Path) -> None:
    """Only document media types can carry the citable paper role."""

    transport = MapTransport({"https://example.org/weights": (200, b"\x00\x01\x02\xff")})
    pack = broker_source_pack(
        [
            {
                "source_id": "blob-paper",
                "kind": "raw-url",
                "url": "https://example.org/weights",
                "requested_role": "paper",
                "basis": "Opaque blob requested as the paper.",
            }
        ],
        broker_dir=tmp_path,
        transport=transport,
    )
    outcome = pack.outcomes[0]
    assert outcome.bound_role == ROLE_DOCUMENTATION
    row = pack.rows[0]
    assert row["broker_role"] == ROLE_DOCUMENTATION
    assert "broker_citable_role" not in row


# -- W-3/D-4: policy refusals are typed, recordable evidence ------------------


def test_http_url_is_typed_unfetchable_by_policy_not_pack_killing(tmp_path: Path) -> None:
    """An http-only reference is recorded per target and never aborts the pack."""

    body = b"class MuRE: pass\n"
    transport = MapTransport(
        {
            COMMITS_URL: (200, json.dumps({"sha": RESOLVED_SHA}).encode()),
            RAW_URL: (200, body),
        }
    )
    pack = broker_source_pack(
        [
            _impl_descriptor(),
            {
                "source_id": "doc-legacy",
                "kind": "raw-url",
                "url": "http://www.cs.toronto.edu/~hinton/absps/nature.pdf",
                "requested_role": "documentation",
                "basis": "Legacy host without TLS.",
            },
        ],
        broker_dir=tmp_path,
        transport=transport,
    )
    outcomes = {item.source_id: item.outcome for item in pack.outcomes}
    assert outcomes == {
        "impl-mure": OUTCOME_FETCHED,
        "doc-legacy": OUTCOME_UNFETCHABLE_BY_POLICY,
    }
    refused = next(item for item in pack.outcomes if item.source_id == "doc-legacy")
    assert "policy" in refused.detail and "https" in refused.detail
    assert refused.sha256 is None
    # The bytes were never contacted, the pack survived, and only the fetched
    # implementation earned a manifest row.
    assert transport.requested == [COMMITS_URL, RAW_URL]
    assert [row["source_id"] for row in pack.rows] == ["impl-mure"]


def test_redirect_to_plaintext_is_refused_even_on_an_allowlisted_host() -> None:
    """The https-only posture holds across redirect hops, not just descriptors."""

    class _Redirecting:
        """Opener stub: first hop answers 302 toward a plaintext location."""

        def open(self, request, timeout):
            raise urllib.error.HTTPError(
                request.full_url,
                302,
                "Found",
                email.message_from_string("Location: http://arxiv.org/abs/1"),
                None,
            )

    transport = UrllibTransport(token_resolver=lambda: None)
    transport._opener = _Redirecting()
    with pytest.raises(RedirectRefused) as refusal:
        transport("https://arxiv.org/abs/1", max_bytes=1024, timeout=1.0)
    assert refusal.value.target == "http://arxiv.org/abs/1"


def test_fixture_transport_round_trip(tmp_path: Path) -> None:
    """The hermetic fixture transport serves the recorded map, nothing else."""

    root = tmp_path / "fixtures"
    root.mkdir()
    (root / "index.json").write_text(
        json.dumps(
            {
                "https://example.org/x": {"status": 200, "body_text": "hello"},
            }
        ),
        encoding="utf-8",
    )
    transport = FixtureTransport(root)
    hit = transport("https://example.org/x", max_bytes=100, timeout=1.0)
    assert hit.status == 200 and hit.body == b"hello"
    miss = transport("https://example.org/y", max_bytes=100, timeout=1.0)
    assert miss.status == 0 and miss.error == "no fixture for url"
