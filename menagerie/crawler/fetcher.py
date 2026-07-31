"""Controlled exact-source fetching into a content-addressed store."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union
from urllib.parse import urlsplit

from enum import StrEnum

from menagerie.crawler.constants import RetrievalStatus
from menagerie.crawler.identity import fsync_directory, hash_bytes, is_sha256, stable_hash

FetchBytes = Callable[[str], bytes]


class ControlledFetchError(RuntimeError):
    """Base class for deterministic controlled-fetch failures."""


class UnpinnedTargetError(ControlledFetchError):
    """Raised when a source target lacks an exact revision or content hash."""


class FetchHashMismatchError(ControlledFetchError):
    """Raised when bytes do not match the pinned digest.

    Retained as the shared base so existing callers keep catching every digest
    disagreement, but nothing should catch it directly any more: the two events
    it used to conflate are structurally unrelated and are now distinct
    subclasses. Telling them apart by parsing the message string was the only
    option before and is never correct.
    """


class CasObjectCorruptError(FetchHashMismatchError):
    """Raised when a local CAS object does not hash to its own address.

    This is our own store failing to match its own content address. Nothing
    upstream can cause it and no retry can fix it: the object on disk is
    damaged. Always fatal.
    """

    def __init__(self, path: Path, expected: str, actual: str) -> None:
        """Bind the damaged object's location and both digests.

        Parameters
        ----------
        path:
            Local CAS object that failed its self-check.
        expected:
            Content address the object is stored under.
        actual:
            Digest the object's bytes actually hash to.
        """

        super().__init__(f"corrupt CAS object {path}: expected {expected}, got {actual}")
        self.path = path
        self.expected_sha256 = expected
        self.actual_sha256 = actual


class UpstreamContentDriftError(FetchHashMismatchError):
    """Raised when a re-fetch of our own earlier fetch returns different bytes.

    This is NOT corruption and NOT substitution. Pages carrying an embedded
    nonce, timestamp, session id, or ad token differ on every retrieval, so at
    campaign scale against live web sources this is routine rather than
    exceptional. It is raised only when the pinned digest is backed by one of
    our own controlled fetches; an author-supplied digest that is unbacked by a
    fetch of ours stays a hard refusal, because there it really is the
    anti-substitution tripwire.

    The retrieved bytes ride along on ``content`` so a caller resolving the
    drift never has to issue a third GET.
    """

    def __init__(
        self,
        *,
        source_id: str,
        url: str,
        expected_sha256: str,
        actual_sha256: str,
        expected_bytes_len: Optional[int],
        actual_bytes_len: int,
        content: bytes,
    ) -> None:
        """Bind both digests, both byte lengths, and the retrieved bytes.

        Parameters
        ----------
        source_id, url:
            Proposal-local source identifier and the exact URL that drifted.
        expected_sha256, actual_sha256:
            Digest of our earlier fetch and of the bytes just retrieved.
        expected_bytes_len:
            Byte length our earlier fetch observed, or ``None`` when the caller
            did not record one. Equal lengths with differing digests is the
            signature of a fixed-width embedded nonce or timestamp.
        actual_bytes_len:
            Byte length just retrieved.
        content:
            The bytes just retrieved, so resolution needs no further request.
        """

        super().__init__(
            f"upstream content drift for {url!r}: expected {expected_sha256} "
            f"({expected_bytes_len} bytes), got {actual_sha256} ({actual_bytes_len} bytes)"
        )
        self.source_id = source_id
        self.url = url
        self.expected_sha256 = expected_sha256
        self.actual_sha256 = actual_sha256
        self.expected_bytes_len = expected_bytes_len
        self.actual_bytes_len = actual_bytes_len
        self.content = content

    def disposition(self) -> dict[str, object]:
        """Return the recorded per-source evidence-integrity disposition.

        Returns
        -------
        dict[str, object]
            Machine-readable record of the drift, carried beside the manifest
            row so an accepted drift is never a silent pass.
        """

        return {
            "event": "upstream-content-drift",
            "source_id": self.source_id,
            "url": self.url,
            "captured_sha256": self.expected_sha256,
            "captured_bytes_len": self.expected_bytes_len,
            "refetched_sha256": self.actual_sha256,
            "refetched_bytes_len": self.actual_bytes_len,
            "equal_length": self.expected_bytes_len == self.actual_bytes_len,
        }


class FetchRetrievalError(ControlledFetchError):
    """Raised when an exact target cannot be retrieved."""


class DigestOrigin(StrEnum):
    """Who the pinned digest on a :class:`FetchTarget` came from.

    The distinction is load-bearing and must never be defaulted away. A digest
    the requester merely *declared* is the anti-substitution tripwire: bytes
    that do not match it are refused outright. A digest one of our own
    controlled fetches *derived* is only a record of what we ourselves saw, so
    a later disagreement is upstream drift rather than a substitution attempt.
    """

    #: Supplied by the requester and unbacked by any fetch of ours. Strict.
    DECLARED = "declared"
    #: Derived by one of our own controlled fetches of the same URL.
    CONTROLLED_FETCH = "controlled-fetch"


@dataclass(frozen=True)
class FetchTarget:
    """One exact controlled-fetch request.

    Parameters
    ----------
    source_id:
        Proposal-local source identifier.
    url:
        Exact HTTPS URL to retrieve.
    revision:
        Exact tag, commit, version, or immutable revision identifier.
    expected_sha256:
        Optional expected content digest, either ``sha256:<64 hex>`` or a bare
        ``<64 hex>`` digest. Empty means the digest is genuinely unknown to the
        requester: the author lane is structurally forbidden from fetching source
        into the campaign, so it can only pin a digest it read from an external
        record (a release manifest, lockfile, or package index). When a digest is
        supplied it is enforced byte-exactly; when it is absent the controlled
        fetch is what learns the digest, and the manifest pins exactly the bytes
        that were retrieved.
    media_type:
        Declared source media type.
    digest_origin:
        Who ``expected_sha256`` came from. Defaults to the strict
        :attr:`DigestOrigin.DECLARED`, so a caller must opt in explicitly before
        a digest is treated as merely our own earlier observation. Defaulting
        the other way would silently disarm the anti-substitution refusal.
    expected_bytes_len:
        Byte length our own earlier fetch observed, when one is known. Carried
        only so a drift report can state both lengths; it is never enforced and
        never substitutes for the digest.
    """

    source_id: str
    url: str
    revision: str
    expected_sha256: str = ""
    media_type: str = "application/octet-stream"
    digest_origin: DigestOrigin = DigestOrigin.DECLARED
    expected_bytes_len: Optional[int] = None


def normalize_expected_sha256(value: Optional[str]) -> str:
    """Normalize a declared content digest to canonical form, or to absence.

    Both spellings the source-request contract advertises are accepted: the
    canonical ``sha256:<64 hex>`` and a bare ``<64 hex>`` digest. Hex case is a
    serialization detail and is folded to lowercase; nothing about the digest
    value itself is relaxed.

    Parameters
    ----------
    value:
        Declared digest, or ``None``/empty when genuinely unknown.

    Returns
    -------
    str
        Canonical ``sha256:<64 lowercase hex>`` digest, or ``""`` for a
        legitimately absent digest.

    Raises
    ------
    UnpinnedTargetError
        If a digest is present but is not a well-formed SHA-256 value.
    """

    candidate = (value or "").strip()
    if not candidate:
        return ""
    digest = candidate[7:] if candidate.lower().startswith("sha256:") else candidate
    if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        raise UnpinnedTargetError(
            "expected_sha256 must be sha256:<64 hex> or <64 hex>, or omitted when unknown; "
            f"got {value!r}"
        )
    return f"sha256:{digest.lower()}"


def cas_path(cas_root: Union[str, Path], content_sha256: str) -> Path:
    """Return the canonical path for one content digest.

    Parameters
    ----------
    cas_root:
        Root of the local source CAS.
    content_sha256:
        Prefixed SHA-256 digest.

    Returns
    -------
    pathlib.Path
        Two-level content-addressed path.

    Raises
    ------
    UnpinnedTargetError
        If the digest is not canonical.
    """

    if not is_sha256(content_sha256):
        raise UnpinnedTargetError("expected_sha256 must be sha256:<64 lowercase hex>")
    digest = content_sha256.removeprefix("sha256:")
    return Path(cas_root) / "sha256" / digest[:2] / digest


def fetch_target(
    target: FetchTarget,
    cas_root: Union[str, Path],
    *,
    fetch_bytes: Optional[FetchBytes] = None,
) -> dict[str, object]:
    """Fetch one exact target and return its hash-bound source manifest.

    A supplied digest is enforced byte-exactly and lets existing correct CAS
    content be reused without a network request. A legitimately absent digest is
    accepted: the controlled fetch is the step that learns the digest, and the
    manifest pins exactly the bytes that were retrieved. In both cases the
    returned ``content_sha256`` is the digest of the verified bytes on disk, so
    every downstream re-verification is unchanged.

    Parameters
    ----------
    target:
        Exact URL, revision, and optional expected digest.
    cas_root:
        Root of the local content-addressed store.
    fetch_bytes:
        Testable byte retriever. The default performs one direct HTTPS GET with
        redirects disabled.

    Returns
    -------
    dict[str, object]
        Hash-bound source manifest.

    Raises
    ------
    UnpinnedTargetError
        If any pin or URL constraint is missing or malformed.
    FetchHashMismatchError
        If existing or retrieved bytes do not match a supplied digest.
    FetchRetrievalError
        If the exact URL cannot be retrieved.
    """

    expected = _validate_target(target, allow_injected_fetch=fetch_bytes is not None)
    if expected:
        destination = cas_path(cas_root, expected)
        if destination.exists():
            content = destination.read_bytes()
            actual = hash_bytes(content)
            if actual != expected:
                raise CasObjectCorruptError(destination, expected, actual)
            return _manifest(
                target, actual, len(content), RetrievalStatus.ALREADY_PRESENT, destination
            )

    retriever = fetch_bytes or _https_get
    try:
        content = retriever(target.url)
    except ControlledFetchError:
        raise
    except Exception as exc:
        raise FetchRetrievalError(f"failed to fetch exact target {target.url!r}: {exc}") from exc
    if not isinstance(content, bytes):
        raise FetchRetrievalError("controlled fetcher must return bytes")
    actual = hash_bytes(content)
    if expected and actual != expected:
        if target.digest_origin is DigestOrigin.CONTROLLED_FETCH:
            # Our own earlier fetch versus our own later fetch. Nothing here is
            # evidence of substitution, so it gets its own type rather than the
            # refusal reserved for an unbacked requester-supplied digest.
            raise UpstreamContentDriftError(
                source_id=target.source_id,
                url=target.url,
                expected_sha256=expected,
                actual_sha256=actual,
                expected_bytes_len=target.expected_bytes_len,
                actual_bytes_len=len(content),
                content=content,
            )
        raise FetchHashMismatchError(
            f"hash mismatch for {target.url!r}: expected {expected}, got {actual}"
        )

    destination = publish_cas_object(cas_root, actual, content)
    return _manifest(target, actual, len(content), RetrievalStatus.FETCHED, destination)


def publish_cas_object(
    cas_root: Union[str, Path],
    content_sha256: str,
    content: bytes,
) -> Path:
    """Atomically publish exact bytes at their canonical content address.

    The single writer for every CAS object, so a controlled fetch, a drift
    resolution, and a promotion from another store all land byte-identically.

    Parameters
    ----------
    cas_root:
        Root of the local source CAS.
    content_sha256:
        Canonical digest the bytes are published under.
    content:
        Exact bytes to publish.

    Returns
    -------
    pathlib.Path
        Canonical CAS object path.

    Raises
    ------
    CasObjectCorruptError
        If the bytes do not hash to the digest they are being published under.
    """

    actual = hash_bytes(content)
    destination = cas_path(cas_root, content_sha256)
    if actual != content_sha256:
        # Refuse to mint an object at an address its own bytes do not have. The
        # promotion path reaches this with bytes from a different store, so this
        # is exactly where a wrong-bytes promotion has to die.
        raise CasObjectCorruptError(destination, content_sha256, actual)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def fetch_targets(
    targets: list[FetchTarget],
    cas_root: Union[str, Path],
    *,
    fetch_bytes: Optional[FetchBytes] = None,
) -> dict[str, object]:
    """Fetch a fixed list of pinned targets and bind their manifests together.

    Parameters
    ----------
    targets:
        Explicit targets; discovery and globbing are intentionally unsupported.
    cas_root:
        Root of the local source CAS.
    fetch_bytes:
        Optional deterministic byte retriever.

    Returns
    -------
    dict[str, object]
        Ordered source manifests plus their aggregate identity.

    Raises
    ------
    CasObjectCorruptError
        If a local CAS object fails its own self-check. Never resolved here.
    FetchHashMismatchError
        If a requester-declared digest unbacked by a fetch of ours is violated.
        Never resolved here either: that is the anti-substitution refusal.
    """

    manifests: list[dict[str, object]] = []
    for target in targets:
        try:
            manifests.append(fetch_target(target, cas_root, fetch_bytes=fetch_bytes))
        except UpstreamContentDriftError as drift:
            # A live page that differs on every retrieval is routine at campaign
            # scale, so this cannot be a lane kill. It is still a real
            # evidence-integrity event, so it is never a silent pass either: the
            # bytes upstream now serves are published, the row re-pins to them so
            # every excerpt locator is re-verified against the NEW bytes, and the
            # drift is recorded per-source beside the row.
            destination = publish_cas_object(cas_root, drift.actual_sha256, drift.content)
            row = _manifest(
                target,
                drift.actual_sha256,
                drift.actual_bytes_len,
                RetrievalStatus.FETCHED,
                destination,
                upstream_drift=drift.disposition(),
            )
            manifests.append(row)
    return {"sources": manifests, "manifest_sha256": stable_hash(manifests)}


def _validate_target(target: FetchTarget, *, allow_injected_fetch: bool) -> str:
    """Validate an exact target before touching the CAS.

    Parameters
    ----------
    target:
        Candidate exact target.
    allow_injected_fetch:
        Whether tests supplied a non-network retriever.

    Returns
    -------
    str
        Canonical expected digest, or ``""`` when the requester legitimately
        does not know it.

    Raises
    ------
    UnpinnedTargetError
        If the target is not exactly identified, its declared digest is
        malformed, or its URL is unsafe.
    """

    if not target.source_id.strip() or not target.revision.strip():
        raise UnpinnedTargetError("source_id and exact revision must be non-empty")
    expected = normalize_expected_sha256(target.expected_sha256)
    parsed = urlsplit(target.url)
    allowed_schemes = {"https"} if not allow_injected_fetch else {"https", "http", "test"}
    if parsed.scheme not in allowed_schemes or not parsed.netloc:
        raise UnpinnedTargetError("controlled fetch requires an exact absolute URL")
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise UnpinnedTargetError("source URLs cannot contain credentials or fragments")
    return expected


def _https_get(url: str) -> bytes:
    """Retrieve one exact HTTPS URL without browser/search behavior.

    Parameters
    ----------
    url:
        Exact URL.

    Returns
    -------
    bytes
        Response body.

    Raises
    ------
    FetchRetrievalError
        If the server redirects or retrieval fails.
    """

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        """Reject redirects so the requested URL remains exact."""

        def redirect_request(
            self,
            req: urllib.request.Request,
            fp: object,
            code: int,
            msg: str,
            headers: object,
            newurl: str,
        ) -> None:
            """Reject an HTTP redirect.

            Parameters
            ----------
            req, fp, code, msg, headers, newurl:
                Redirect callback values supplied by urllib.

            Returns
            -------
            None
                Redirects are always rejected.
            """

            return None

    request = urllib.request.Request(url, headers={"User-Agent": "torchlens-menagerie-fetcher/2"})
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=30) as response:
            return response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise FetchRetrievalError(str(exc)) from exc


def _manifest(
    target: FetchTarget,
    content_sha256: str,
    length: int,
    status: RetrievalStatus,
    path: Path,
    *,
    upstream_drift: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    """Build one deterministic source manifest.

    Parameters
    ----------
    target:
        Exact source target.
    content_sha256:
        Digest of the exact verified bytes now in the CAS. This is the pin every
        downstream consumer re-verifies, whether the requester declared it or the
        controlled fetch learned it.
    length:
        Verified byte length.
    status:
        Fetch or reuse outcome.
    path:
        Local CAS object path.
    upstream_drift:
        Recorded evidence-integrity disposition when this row re-pinned to bytes
        that moved under an earlier fetch of ours. Absent on every ordinary row,
        so its mere presence is the signal.

    Returns
    -------
    dict[str, object]
        Manifest bound to the exact bytes.
    """

    body: dict[str, object] = {
        "source_id": target.source_id,
        "url": target.url,
        "revision": target.revision,
        "content_sha256": content_sha256,
        "fetched_bytes_len": length,
        # Deliberately still `fetched`: these bytes really were fetched in this
        # pass and really are the CAS object this row names. Minting a third
        # status here would drop the row out of the `{fetched, already-present}`
        # gates that admit a source for excerpt verification, which would turn
        # "we noticed drift" into "we silently stopped checking the excerpt" --
        # the exact opposite of the point.
        "retrieval_status": status.value,
        "media_type": target.media_type,
        "cas_path": str(path),
    }
    if upstream_drift is not None:
        body["upstream_drift"] = upstream_drift
    return {**body, "manifest_sha256": stable_hash(body)}
