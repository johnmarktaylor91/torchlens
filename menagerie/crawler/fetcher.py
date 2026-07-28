"""Controlled exact-source fetching into a content-addressed store."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union
from urllib.parse import urlsplit

from menagerie.crawler.constants import RetrievalStatus
from menagerie.crawler.identity import fsync_directory, hash_bytes, is_sha256, stable_hash

FetchBytes = Callable[[str], bytes]


class ControlledFetchError(RuntimeError):
    """Base class for deterministic controlled-fetch failures."""


class UnpinnedTargetError(ControlledFetchError):
    """Raised when a source target lacks an exact revision or content hash."""


class FetchHashMismatchError(ControlledFetchError):
    """Raised when retrieved bytes do not match the pinned digest."""


class FetchRetrievalError(ControlledFetchError):
    """Raised when an exact target cannot be retrieved."""


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
    """

    source_id: str
    url: str
    revision: str
    expected_sha256: str = ""
    media_type: str = "application/octet-stream"


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
                raise FetchHashMismatchError(
                    f"corrupt CAS object {destination}: expected {expected}, got {actual}"
                )
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
        raise FetchHashMismatchError(
            f"hash mismatch for {target.url!r}: expected {expected}, got {actual}"
        )

    destination = cas_path(cas_root, actual)
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
    return _manifest(target, actual, len(content), RetrievalStatus.FETCHED, destination)


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
    """

    manifests = [fetch_target(target, cas_root, fetch_bytes=fetch_bytes) for target in targets]
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
        "retrieval_status": status.value,
        "media_type": target.media_type,
        "cas_path": str(path),
    }
    return {**body, "manifest_sha256": stable_hash(body)}
