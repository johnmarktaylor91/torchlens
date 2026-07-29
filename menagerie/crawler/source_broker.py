"""Machine-owned source broker: where exact strings come from.

The LLM **requests objects; the broker produces strings.** Stage 1 names a
repository + path + confirmed tag/branch (or a direct URL, or a paper); this
module resolves refs to immutable commit SHAs via the forge API, fetches bytes
through a bounded streaming transport with an allowlisted redirect policy,
derives citation metadata from authoritative registries (arXiv, Crossref,
OpenReview), and binds source roles. Every exact identifier in the resulting
manifest rows — commit SHA, content digest, citation fields — is derived by
this code from a recorded receipt, never taken from the model. This is what
kills the fabricated-SHA class structurally instead of detecting it after the
fact.

Per-target outcomes are independent and typed
(``fetched | oversized | unreachable | redirect-refused | rate-limited |
bad-ref``): one dead link never aborts the pack, and every failure carries its
receipt so stage 2 and the validators see exactly what exists.

Two of those outcomes are claims about *whose* failure it was, and the
distinction is load-bearing. ``bad-ref`` says the author supplied a reference
that does not exist. ``rate-limited`` says the reference was fine and the forge
throttled *our* client. Recording the second as the first is a false statement
in a durable record; it also burns retries "repairing" a reference that was
never wrong, and it would mislead any later triage of what actually needs work.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Union

from menagerie.crawler.identity import hash_bytes, utc_now
from menagerie.crawler.models import JsonObject

BROKER_PACK_VERSION = "menagerie.crawler.source-broker-pack.v1"

#: Hosts a redirect may land on. Any other redirect target is a typed refusal.
REDIRECT_HOST_ALLOWLIST = frozenset(
    {
        "doi.org",
        "dx.doi.org",
        "arxiv.org",
        "export.arxiv.org",
        "openreview.net",
        "api.openreview.net",
        "github.com",
        "api.github.com",
        "raw.githubusercontent.com",
        "objects.githubusercontent.com",
        "codeload.github.com",
        "zenodo.org",
        "gitlab.com",
    }
)

#: Closed per-target outcome vocabulary.
OUTCOME_FETCHED = "fetched"
OUTCOME_OVERSIZED = "oversized"
OUTCOME_UNREACHABLE = "unreachable"
OUTCOME_REDIRECT_REFUSED = "redirect-refused"
OUTCOME_BAD_REF = "bad-ref"
#: The forge throttled us. Attributable to the crawler, never to the author, and
#: retryable once the window resets.
OUTCOME_RATE_LIMITED = "rate-limited"
#: A real reference on a forge this resolver cannot yet address. A limitation of
#: ours, not a defect in what the author supplied.
OUTCOME_UNSUPPORTED_FORGE = "unsupported-forge"
OUTCOME_PAPER_DERIVATION_ONLY = "paper-derivation-only"
OUTCOME_PROBED = "probed"

#: Outcomes that describe a failure of ours or of the forge rather than a defect
#: in the reference the author supplied. Nothing in this set justifies asking an
#: author to fix its reference.
NON_AUTHOR_FAULT_OUTCOMES = frozenset(
    {
        OUTCOME_RATE_LIMITED,
        OUTCOME_UNREACHABLE,
        OUTCOME_OVERSIZED,
        OUTCOME_UNSUPPORTED_FORGE,
    }
)

#: Closed bound-role vocabulary. ``introducing-paper`` is broker-assigned only.
ROLE_IMPLEMENTATION = "implementation"
ROLE_INTRODUCING_PAPER = "introducing-paper"
ROLE_DOCUMENTATION = "documentation"
ROLE_PROBE = "probe"

DEFAULT_TARGET_BYTE_CEILING = 8 * 1024 * 1024
DEFAULT_TOTAL_BYTE_CEILING = 64 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_REDIRECTS = 5

_ARXIV_ID_PATTERN = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|^)(\d{4}\.\d{4,5})(?:v\d+)?")
_DOI_PATTERN = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+?)(?:[.,;]?(?:\s|$))")
_OPENREVIEW_PATTERN = re.compile(r"openreview\.net/(?:forum|pdf)\?id=([A-Za-z0-9_-]+)")
_SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_GITHUB_REPO_PATTERN = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)

_MEDIA_TYPES = {
    ".py": "text/x-python",
    ".md": "text/markdown",
    ".rst": "text/x-rst",
    ".txt": "text/plain",
    ".json": "application/json",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".html": "text/html",
    ".pdf": "application/pdf",
    ".lua": "text/x-lua",
    ".m": "text/x-matlab",
    ".c": "text/x-c",
    ".cc": "text/x-c++",
    ".cpp": "text/x-c++",
    ".cu": "text/x-cuda",
    ".proto": "text/x-protobuf",
    ".cfg": "text/plain",
    ".prototxt": "text/plain",
}
_IMPLEMENTATION_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cfg",
        ".cpp",
        ".cu",
        ".json",
        ".lua",
        ".m",
        ".proto",
        ".prototxt",
        ".py",
        ".yaml",
        ".yml",
    }
)


#: Hosts the GitHub credential may be sent to. Deliberately just the API: the raw
#: and codeload hosts serve public bytes without a credential, and every extra
#: host a bearer token is offered to is another way for it to escape.
GITHUB_API_HOST = "api.github.com"

#: Environment names carrying a GitHub credential, in resolution order.
#: ``GH_TOKEN`` precedes ``GITHUB_TOKEN`` because that is ``gh``'s own documented
#: precedence, and ``gh auth token`` is our fallback: if the two disagreed we
#: would authenticate as a different identity than the CLI an operator debugs
#: with, which is the kind of divergence nobody thinks to check.
GITHUB_TOKEN_ENV_NAMES = ("GH_TOKEN", "GITHUB_TOKEN")

#: Pinned API version, so a future default shift cannot silently change parsing.
GITHUB_API_VERSION = "2022-11-28"

#: Statuses a forge uses to throttle. ``429`` *is* "too many requests" by
#: definition, so it needs no corroboration; ``403`` is ambiguous -- genuinely
#: forbidden and rate limited share it -- so it is only read as throttling when
#: the response's own rate-limit headers say so.
RATE_LIMIT_STATUSES = frozenset({403, 429})

#: Refs the forge answers authoritatively: the ref genuinely does not exist.
BAD_REF_STATUSES = frozenset({404, 422})

_GH_TOKEN_COMMAND = ("gh", "auth", "token")
_GH_TOKEN_TIMEOUT_SECONDS = 10.0


class SourceBrokerError(ValueError):
    """Raised when a discovery descriptor cannot be brokered at all."""


def resolve_github_token() -> Optional[str]:
    """Return a GitHub credential when one is available, else ``None``.

    Unauthenticated ``api.github.com`` is capped at 60 requests per hour, which
    cannot support a 28,482-model campaign resolving at least one ref each. The
    cap was hit in a live rung on 2026-07-29: the forge answered ``403``, the
    reference was perfectly good, and the model was blamed for it.
    Authenticated, the same endpoint allows 5,000 requests per hour.

    Resolution order is ``GH_TOKEN``, ``GITHUB_TOKEN``, then ``gh auth token``.
    The environment comes first because it is the deliberate operator override
    and costs nothing to read; ``gh`` is last because it spawns a subprocess and
    depends on an interactive login that may not exist under a supervisor. Among
    the two variables ``GH_TOKEN`` wins, matching ``gh``'s own precedence, so the
    broker and the CLI an operator debugs with never authenticate as different
    identities.

    An absent credential returns ``None`` and the caller proceeds
    unauthenticated -- degraded exactly as before, never refusing to start. This
    mirrors :func:`menagerie.crawler.author_executor.exa_mcp_config`: a missing
    credential must not be why a campaign cannot run.

    Returns
    -------
    str | None
        The credential, or ``None`` when no source supplied one.
    """

    for name in GITHUB_TOKEN_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        completed = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            _GH_TOKEN_COMMAND,
            capture_output=True,
            text=True,
            timeout=_GH_TOKEN_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # `gh` absent, unrunnable, or hung. Never fatal: this is the fallback.
        return None
    if completed.returncode != 0:
        # Deliberately discards stderr rather than recording it. `gh` failure
        # text is not diagnostic enough to be worth the risk of a credential
        # fragment reaching a log in a public repository.
        return None
    token = completed.stdout.strip()
    return token or None


@dataclass(frozen=True)
class RateLimitSignal:
    """Machine-read evidence that a response was throttled rather than refused.

    Parameters
    ----------
    status:
        The observed HTTP status.
    remaining:
        ``x-ratelimit-remaining``, when the forge reported it.
    retry_after_seconds:
        ``retry-after`` in seconds, when the forge reported it.
    reset_epoch:
        ``x-ratelimit-reset`` as a Unix timestamp, when the forge reported it.
    resource:
        ``x-ratelimit-resource``, naming which budget was exhausted.
    """

    status: int
    remaining: Optional[int] = None
    retry_after_seconds: Optional[float] = None
    reset_epoch: Optional[int] = None
    resource: Optional[str] = None

    def to_dict(self) -> JsonObject:
        """Return the JSON receipt fragment."""

        return {
            "status": self.status,
            "remaining": self.remaining,
            "retry_after_seconds": self.retry_after_seconds,
            "reset_epoch": self.reset_epoch,
            "resource": self.resource,
        }


def _header(headers: Mapping[str, str], name: str) -> Optional[str]:
    """Return one header case-insensitively, or ``None``."""

    for key, value in headers.items():
        if key.lower() == name:
            text = str(value).strip()
            return text or None
    return None


def _int_header(headers: Mapping[str, str], name: str) -> Optional[int]:
    """Return one integral header value, or ``None`` when absent or unparseable."""

    raw = _header(headers, name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def rate_limit_signal(status: int, headers: Mapping[str, str]) -> Optional[RateLimitSignal]:
    """Classify one response as throttled, using the forge's own headers.

    A ``403`` is deliberately NOT read as throttling on its own: a private
    repository, a blocked client, and an exhausted quota all share that status,
    and guessing would relabel a genuine refusal as a retryable wait. It counts
    only when the response says ``x-ratelimit-remaining: 0`` (GitHub's primary
    limit) or carries ``retry-after`` (its secondary limit). ``429`` is
    unambiguous on its own.

    Parameters
    ----------
    status:
        Final HTTP status.
    headers:
        Final response headers.

    Returns
    -------
    RateLimitSignal | None
        The signal when the response was throttled, else ``None``.
    """

    if status not in RATE_LIMIT_STATUSES:
        return None
    remaining = _int_header(headers, "x-ratelimit-remaining")
    retry_after = _int_header(headers, "retry-after")
    reset = _int_header(headers, "x-ratelimit-reset")
    throttled = status == 429 or remaining == 0 or retry_after is not None
    if not throttled:
        return None
    return RateLimitSignal(
        status=status,
        remaining=remaining,
        retry_after_seconds=float(retry_after) if retry_after is not None else None,
        reset_epoch=reset,
        resource=_header(headers, "x-ratelimit-resource"),
    )


@dataclass(frozen=True)
class TransportResponse:
    """One transport round trip, with its full redirect chain.

    Parameters
    ----------
    status:
        Final HTTP status; ``0`` when the transport never reached a server.
    final_url:
        URL that produced the final response.
    redirect_chain:
        Every URL visited, in order, including the first.
    body:
        Response bytes, possibly truncated at the byte ceiling.
    truncated:
        Whether the byte ceiling cut the body short.
    error:
        Transport-level failure description, or ``None``.
    headers:
        Final response headers. These are what let a throttled response be told
        apart from a genuinely forbidden one instead of guessed at from status.
    """

    status: int
    final_url: str
    redirect_chain: tuple[str, ...]
    body: bytes
    truncated: bool
    error: Optional[str] = None
    headers: Mapping[str, str] = field(default_factory=dict)


class Transport(Protocol):
    """Bounded byte transport with recorded redirects."""

    def __call__(self, url: str, *, max_bytes: int, timeout: float) -> TransportResponse:
        """Fetch one URL within the ceiling, recording every redirect hop."""
        ...


class RedirectRefused(Exception):
    """Internal signal: a redirect targeted a host outside the allowlist."""

    def __init__(self, chain: tuple[str, ...], target: str):
        super().__init__(f"redirect to non-allowlisted host: {target}")
        self.chain = chain
        self.target = target


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Redirect handler that surfaces 3xx responses instead of following them."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: Any, headers: Any, newurl: str):
        """Refuse automatic redirects so each hop is policy-checked."""

        return None


class UrllibTransport:
    """Streaming HTTPS transport with per-hop redirect policy and byte ceiling.

    The GitHub credential, when one exists, is attached per hop and only for
    :data:`GITHUB_API_HOST`. Attaching it once for the whole call would send it
    onward through any allowlisted redirect that leaves the API host, so the
    decision is re-made against the URL actually being requested.
    """

    def __init__(
        self,
        *,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        redirect_allowlist: frozenset[str] = REDIRECT_HOST_ALLOWLIST,
        user_agent: str = "menagerie-crawler-source-broker/1",
        token_resolver: Callable[[], Optional[str]] = resolve_github_token,
    ) -> None:
        self.max_redirects = max_redirects
        self.redirect_allowlist = redirect_allowlist
        self.user_agent = user_agent
        self._token_resolver = token_resolver
        self._token: Optional[str] = None
        self._token_resolved = False
        self._opener = urllib.request.build_opener(_NoRedirect())

    def _github_token(self) -> Optional[str]:
        """Resolve the credential once per transport, and only when needed.

        Memoized so a pack resolving many refs pays at most one ``gh``
        subprocess, and lazy so a pack that never touches the forge API pays
        none at all.
        """

        if not self._token_resolved:
            self._token = self._token_resolver()
            self._token_resolved = True
        return self._token

    def github_credential_mode(self) -> str:
        """Report whether forge calls are authenticated, never the credential."""

        return "authenticated" if self._github_token() else "anonymous"

    def _request_headers(self, url: str) -> dict[str, str]:
        """Build the headers for one hop, credential included only for the API."""

        headers = {"User-Agent": self.user_agent, "Accept": "*/*"}
        if (urllib.parse.urlsplit(url).hostname or "").lower() != GITHUB_API_HOST:
            return headers
        token = self._github_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = GITHUB_API_VERSION
        return headers

    def __call__(self, url: str, *, max_bytes: int, timeout: float) -> TransportResponse:
        """Fetch one URL, following only allowlisted redirects, streaming-bounded."""

        chain: list[str] = [url]
        current = url
        for _hop in range(self.max_redirects + 1):
            request = urllib.request.Request(current, headers=self._request_headers(current))
            try:
                with self._opener.open(request, timeout=timeout) as response:
                    body, truncated = _read_bounded(response, max_bytes)
                    return TransportResponse(
                        status=int(getattr(response, "status", 200) or 200),
                        final_url=current,
                        redirect_chain=tuple(chain),
                        body=body,
                        truncated=truncated,
                        headers=_response_headers(getattr(response, "headers", None)),
                    )
            except urllib.error.HTTPError as exc:
                if exc.code in (301, 302, 303, 307, 308):
                    location = exc.headers.get("Location") if exc.headers else None
                    if not location:
                        return TransportResponse(
                            status=exc.code,
                            final_url=current,
                            redirect_chain=tuple(chain),
                            body=b"",
                            truncated=False,
                            error="redirect without Location",
                            headers=_response_headers(exc.headers),
                        )
                    target = urllib.parse.urljoin(current, location)
                    host = urllib.parse.urlsplit(target).hostname or ""
                    if host not in self.redirect_allowlist:
                        raise RedirectRefused(tuple([*chain, target]), target) from exc
                    chain.append(target)
                    current = target
                    continue
                body = exc.read() if hasattr(exc, "read") else b""
                return TransportResponse(
                    status=exc.code,
                    final_url=current,
                    redirect_chain=tuple(chain),
                    body=body[:max_bytes],
                    truncated=False,
                    error=f"http {exc.code}",
                    headers=_response_headers(exc.headers),
                )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                return TransportResponse(
                    status=0,
                    final_url=current,
                    redirect_chain=tuple(chain),
                    body=b"",
                    truncated=False,
                    error=str(exc),
                )
        return TransportResponse(
            status=0,
            final_url=current,
            redirect_chain=tuple(chain),
            body=b"",
            truncated=False,
            error="too many redirects",
        )


def _response_headers(headers: Any) -> dict[str, str]:
    """Normalize a response's headers to a lowercase-keyed mapping.

    Only the small set of rate-limit headers is retained. A response's headers
    are copied into receipts, and receipts are committed to a public repository,
    so this is an allowlist rather than a filter: nothing unanticipated can ride
    along from a forge response into a durable artifact.
    """

    if headers is None:
        return {}
    retained = (
        "retry-after",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "x-ratelimit-resource",
        "x-ratelimit-used",
    )
    items: dict[str, str] = {}
    for name in retained:
        try:
            value = headers.get(name)
        except AttributeError:  # pragma: no cover -- defensive
            return {}
        if value is not None:
            items[name] = str(value).strip()
    return items


def _read_bounded(response: Any, max_bytes: int) -> tuple[bytes, bool]:
    """Stream a response body up to the ceiling, reporting truncation."""

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(65536)
        if not chunk:
            return b"".join(chunks), False
        total += len(chunk)
        if total > max_bytes:
            chunks.append(chunk[: max_bytes - (total - len(chunk))])
            return b"".join(chunks), True
        chunks.append(chunk)


class FixtureTransport:
    """Hermetic transport for tests and dry runs: URLs resolve from a fixture map.

    The fixture root holds ``index.json`` mapping URLs to
    ``{"status", "body_text" | "body_file", "redirect_chain", "final_url",
    "truncated", "error", "headers"}``. An unmapped URL is unreachable — the fixture
    transport never touches the network, so a test can prove exactly which
    URLs the broker asked for.
    """

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        index_path = self.root / "index.json"
        try:
            self.index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceBrokerError(f"broker fixture index unreadable: {exc}") from exc
        self.requested: list[str] = []

    def __call__(self, url: str, *, max_bytes: int, timeout: float) -> TransportResponse:
        """Resolve one URL from the fixture map."""

        del timeout
        self.requested.append(url)
        entry = self.index.get(url)
        if not isinstance(entry, Mapping):
            return TransportResponse(
                status=0,
                final_url=url,
                redirect_chain=(url,),
                body=b"",
                truncated=False,
                error="no fixture for url",
            )
        if isinstance(entry.get("body_file"), str):
            body = (self.root / str(entry["body_file"])).read_bytes()
        else:
            body = str(entry.get("body_text", "")).encode("utf-8")
        truncated = bool(entry.get("truncated", False)) or len(body) > max_bytes
        if len(body) > max_bytes:
            body = body[:max_bytes]
        chain = tuple(entry.get("redirect_chain", (url,)))
        target = str(entry.get("refused_redirect_target", ""))
        if target:
            raise RedirectRefused(chain, target)
        raw_headers = entry.get("headers")
        headers = (
            {str(key).lower(): str(value) for key, value in raw_headers.items()}
            if isinstance(raw_headers, Mapping)
            else {}
        )
        return TransportResponse(
            status=int(entry.get("status", 200)),
            final_url=str(entry.get("final_url", url)),
            redirect_chain=chain,
            body=body,
            truncated=truncated,
            error=(str(entry["error"]) if entry.get("error") else None),
            headers=headers,
        )


def default_transport() -> Transport:
    """Return the configured transport.

    ``MENAGERIE_BROKER_FIXTURES`` selects the hermetic fixture transport for
    tests and dry runs; production uses the bounded urllib transport.
    """

    fixtures = os.environ.get("MENAGERIE_BROKER_FIXTURES")
    if fixtures:
        return FixtureTransport(fixtures)
    return UrllibTransport()


@dataclass(frozen=True)
class BrokerOutcome:
    """One target's complete, typed broker outcome with receipts."""

    source_id: str
    kind: str
    requested_role: str
    bound_role: Optional[str]
    outcome: str
    url: Optional[str]
    final_url: Optional[str]
    redirect_chain: tuple[str, ...]
    status: Optional[int]
    bytes_fetched: int
    sha256: Optional[str]
    media_type: Optional[str]
    media_type_method: Optional[str]
    resolver_receipt: Optional[JsonObject]
    derived_citation: Optional[JsonObject]
    detail: str = ""
    #: Throttling evidence when ``outcome`` is ``rate-limited``. Present so a
    #: consumer can act on the forge's own reset instant instead of guessing.
    rate_limit: Optional[JsonObject] = None

    @property
    def rate_limit_retry_after(self) -> Optional[float]:
        """Return the requested wait in seconds, when the forge named one."""

        value = (self.rate_limit or {}).get("retry_after_seconds")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def rate_limit_reset_epoch(self) -> Optional[int]:
        """Return the reset instant as a Unix timestamp, when the forge named one."""

        value = (self.rate_limit or {}).get("reset_epoch")
        return int(value) if isinstance(value, int) else None

    def to_dict(self) -> JsonObject:
        """Return the JSON diagnostic row."""

        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "requested_role": self.requested_role,
            "bound_role": self.bound_role,
            "outcome": self.outcome,
            "url": self.url,
            "final_url": self.final_url,
            "redirect_chain": list(self.redirect_chain),
            "status": self.status,
            "bytes_fetched": self.bytes_fetched,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "media_type_method": self.media_type_method,
            "resolver_receipt": self.resolver_receipt,
            "derived_citation": self.derived_citation,
            "detail": self.detail,
            "rate_limit": self.rate_limit,
        }


@dataclass
class BrokerPack:
    """The frozen product of one broker pass."""

    rows: list[JsonObject] = field(default_factory=list)
    outcomes: list[BrokerOutcome] = field(default_factory=list)
    derived_citations: list[JsonObject] = field(default_factory=list)
    total_bytes: int = 0

    def implementation_rows(self) -> list[JsonObject]:
        """Return the manifest rows bound to the implementation role."""

        return [row for row in self.rows if row.get("broker_role") == ROLE_IMPLEMENTATION]

    def rate_limited_outcomes(self) -> list[BrokerOutcome]:
        """Return the targets the forge threw us out of."""

        return [item for item in self.outcomes if item.outcome == OUTCOME_RATE_LIMITED]

    def blocked_by_rate_limit(self) -> bool:
        """Report whether throttling, not a bad reference, emptied this pack.

        ``True`` means the pack produced no implementation row AND at least one
        target was throttled -- the model is not unauthorable, we were simply
        not allowed to look. Callers use this to keep the failure attributed to
        the campaign instead of the author.
        """

        return not self.implementation_rows() and bool(self.rate_limited_outcomes())

    def retry_after_seconds(self) -> Optional[float]:
        """Return the longest wait any throttled target asked for, if any."""

        waits = [
            item.rate_limit_retry_after
            for item in self.rate_limited_outcomes()
            if item.rate_limit_retry_after is not None
        ]
        return max(waits) if waits else None

    def rate_limit_reset_epoch(self) -> Optional[int]:
        """Return the latest reset instant any throttled target reported, if any."""

        resets = [
            item.rate_limit_reset_epoch
            for item in self.rate_limited_outcomes()
            if item.rate_limit_reset_epoch is not None
        ]
        return max(resets) if resets else None

    def to_dict(self) -> JsonObject:
        """Return the JSON pack: lane rows plus complete broker diagnostics."""

        return {
            "pack_version": BROKER_PACK_VERSION,
            "sources": self.rows,
            "broker": {
                "outcomes": [outcome.to_dict() for outcome in self.outcomes],
                "derived_citations": self.derived_citations,
                "total_bytes": self.total_bytes,
            },
        }


def broker_source_pack(
    descriptors: Sequence[Mapping[str, Any]],
    *,
    broker_dir: Union[str, Path],
    transport: Optional[Transport] = None,
    clock: Callable[[], str] = utc_now,
    target_byte_ceiling: int = DEFAULT_TARGET_BYTE_CEILING,
    total_byte_ceiling: int = DEFAULT_TOTAL_BYTE_CEILING,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> BrokerPack:
    """Broker one stage-1 ``FOUND`` source list into machine-derived rows.

    Parameters
    ----------
    descriptors:
        Untrusted model-emitted source descriptors. Each names an object
        (``forge-file``: repo + path + ref; ``raw-url``: url; ``paper``: url or
        identifier); none may carry the exact strings the broker derives.
    broker_dir:
        Attempt-local directory receiving raw evidence bytes and receipts.
    transport:
        Bounded transport; defaults to :func:`default_transport`.
    clock:
        Injectable timestamp source for receipts.
    target_byte_ceiling, total_byte_ceiling, timeout_seconds:
        Transport budgets. Exceeding the per-target ceiling is ``oversized``;
        the total ceiling stops further fetches with typed outcomes.

    Returns
    -------
    BrokerPack
        Lane-shaped manifest rows (only for successfully fetched targets) plus
        per-target diagnostics, receipts, and derived citations.
    """

    transport = transport or default_transport()
    broker_dir = Path(broker_dir)
    evidence_dir = broker_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    pack = BrokerPack()
    source_ids: set[str] = set()
    for position, raw in enumerate(descriptors):
        descriptor = _validated_descriptor(raw, position)
        source_id = str(descriptor["source_id"])
        if source_id in source_ids:
            raise SourceBrokerError(f"source descriptor {position} duplicates source_id {source_id!r}")
        source_ids.add(source_id)
        if pack.total_bytes >= total_byte_ceiling:
            pack.outcomes.append(
                _failure_outcome(
                    descriptor,
                    OUTCOME_UNREACHABLE,
                    detail="total broker byte ceiling exhausted before this target",
                )
            )
            continue
        remaining = min(target_byte_ceiling, total_byte_ceiling - pack.total_bytes)
        kind = descriptor["kind"]
        if kind == "forge-file":
            outcome = _broker_forge_file(
                descriptor,
                transport=transport,
                evidence_dir=evidence_dir,
                clock=clock,
                max_bytes=remaining,
                timeout=timeout_seconds,
            )
        elif kind == "raw-url":
            outcome = _broker_raw_url(
                descriptor,
                transport=transport,
                evidence_dir=evidence_dir,
                clock=clock,
                max_bytes=remaining,
                timeout=timeout_seconds,
            )
        else:  # paper
            outcome = _broker_paper(
                descriptor,
                transport=transport,
                evidence_dir=evidence_dir,
                clock=clock,
                max_bytes=remaining,
                timeout=timeout_seconds,
            )
        pack.outcomes.append(outcome)
        pack.total_bytes += outcome.bytes_fetched
        if outcome.derived_citation is not None:
            pack.derived_citations.append(outcome.derived_citation)
        if outcome.outcome == OUTCOME_FETCHED and outcome.bound_role in (
            ROLE_IMPLEMENTATION,
            ROLE_DOCUMENTATION,
        ):
            pack.rows.append(_manifest_row(descriptor, outcome))
    return pack


def write_broker_outputs(pack: BrokerPack, broker_dir: Union[str, Path]) -> Path:
    """Persist the pack's receipts next to its evidence bytes.

    Returns
    -------
    pathlib.Path
        The receipts path.
    """

    broker_dir = Path(broker_dir)
    broker_dir.mkdir(parents=True, exist_ok=True)
    path = broker_dir / "receipts.json"
    payload = pack.to_dict()
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    _refuse_credential_bearing_locators(data)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


#: URL shapes that carry a credential: userinfo (``https://user:secret@host``)
#: or a token-bearing query parameter. The broker authenticates with a request
#: header precisely so nothing like this can exist, and this pattern is the
#: tripwire that keeps it true if someone later takes the easier route.
#:
#: The query-parameter list is deliberately confined to names that are only ever
#: credentials. A bare ``token=`` or ``key=`` is left out on purpose: those do
#: appear in legitimate signed links an author may cite, and refusing the whole
#: pack over one would trade a real leak guard for a self-inflicted outage.
_CREDENTIAL_BEARING_LOCATOR = re.compile(
    r"https?://[^/\s\"]*:[^/\s\"]*@"
    r"|[?&](?:access_token|api_key|apikey|x-api-key|exaapikey|private_token|"
    r"client_secret|password)=[^&\s\"]+",
    re.IGNORECASE,
)


def _refuse_credential_bearing_locators(serialized: str) -> None:
    """Refuse to persist any receipt payload carrying a credential in a locator.

    ``johnmarktaylor91/torchlens`` is a public repository and broker receipts are
    durable. A credential that reached an artifact could not be unpublished, so
    this fails the write rather than emitting it -- the same fail-loud posture
    the rest of the broker takes.

    Raises
    ------
    SourceBrokerError
        When a credential-bearing locator is present. The offending value is
        deliberately NOT included in the message.
    """

    if _CREDENTIAL_BEARING_LOCATOR.search(serialized) is not None:
        raise SourceBrokerError(
            "broker receipts carry a credential-bearing locator; refusing to persist "
            "(the forge credential belongs in a request header, never in a URL)"
        )


# -- descriptor validation -------------------------------------------------


_DESCRIPTOR_KINDS = frozenset({"forge-file", "raw-url", "paper"})
_REQUESTED_ROLES = frozenset({"implementation", "paper", "documentation", "probe"})
_COMMON_DESCRIPTOR_FIELDS = frozenset(
    {"source_id", "kind", "requested_role", "media_type_hint", "basis"}
)
_DESCRIPTOR_FIELDS = {
    "forge-file": _COMMON_DESCRIPTOR_FIELDS | {"repo", "path", "ref"},
    "raw-url": _COMMON_DESCRIPTOR_FIELDS | {"url"},
    "paper": _COMMON_DESCRIPTOR_FIELDS | {"url", "identifier"},
}
_MACHINE_OWNED_DESCRIPTOR_FIELDS = frozenset(
    {
        "revision",
        "commit_sha",
        "expected_sha256",
        "content_sha256",
        "sha256",
        "final_url",
        "redirect_chain",
        "resolver_receipt",
        "broker_role",
        "media_type",
        "derived_citation",
    }
)


def _validated_descriptor(raw: Mapping[str, Any], position: int) -> JsonObject:
    """Validate one untrusted descriptor into the closed broker shape.

    Raises
    ------
    SourceBrokerError
        When the descriptor is structurally unusable, names an unsupported
        kind or role, or smuggles machine-owned exact strings.
    """

    if not isinstance(raw, Mapping):
        raise SourceBrokerError(f"source descriptor {position} must be an object")
    kind = str(raw.get("kind", ""))
    if kind not in _DESCRIPTOR_KINDS:
        raise SourceBrokerError(f"source descriptor {position} kind {kind!r} is not supported")
    role = str(raw.get("requested_role", ""))
    if role not in _REQUESTED_ROLES:
        raise SourceBrokerError(
            f"source descriptor {position} requested_role {role!r} is not supported"
        )
    source_id = str(raw.get("source_id", "")).strip()
    if _SOURCE_ID_PATTERN.fullmatch(source_id) is None:
        raise SourceBrokerError(
            f"source descriptor {position} source_id must match "
            "^[a-z0-9][a-z0-9._-]{0,63}$"
        )
    for forbidden in _MACHINE_OWNED_DESCRIPTOR_FIELDS:
        if forbidden in raw:
            raise SourceBrokerError(
                f"source descriptor {position} carries machine-owned field {forbidden!r}; "
                "exact strings are derived by the broker, never authored"
            )
    unknown = set(raw) - _DESCRIPTOR_FIELDS[kind]
    if unknown:
        raise SourceBrokerError(
            f"source descriptor {position} carries unsupported fields {sorted(unknown)!r}"
        )
    basis = str(raw.get("basis", "")).strip()
    if not basis:
        raise SourceBrokerError(f"source descriptor {position} must carry a nonempty basis")
    descriptor: JsonObject = {
        "source_id": source_id,
        "kind": kind,
        "requested_role": role,
        "basis": basis,
        "repo": str(raw.get("repo", "")).strip(),
        "path": str(raw.get("path", "")).strip(),
        "ref": str(raw.get("ref", "")).strip(),
        "url": str(raw.get("url", "")).strip(),
        "identifier": str(raw.get("identifier", "")).strip(),
        "media_type_hint": str(raw.get("media_type_hint", "")).strip(),
    }
    if kind == "forge-file" and not (
        descriptor["repo"] and descriptor["path"] and descriptor["ref"]
    ):
        raise SourceBrokerError(
            f"forge-file descriptor {position} must name repo, path, and a confirmed ref"
        )
    if kind == "raw-url" and not descriptor["url"]:
        raise SourceBrokerError(f"raw-url descriptor {position} must name a url")
    if kind == "paper" and not (descriptor["url"] or descriptor["identifier"]):
        raise SourceBrokerError(f"paper descriptor {position} must name a url or identifier")
    if descriptor["url"] and urllib.parse.urlsplit(str(descriptor["url"])).scheme != "https":
        raise SourceBrokerError(f"source descriptor {position} url must use https")
    if kind == "forge-file":
        path = str(descriptor["path"])
        normalized = PurePosixPath(path)
        if (
            normalized.is_absolute()
            or ".." in normalized.parts
            or "." in normalized.parts
            or "\\" in path
            or str(normalized) != path
        ):
            raise SourceBrokerError(
                f"forge-file descriptor {position} path must be normalized and relative"
            )
    return descriptor


def _derive_media_type(name: str, body: bytes) -> tuple[str, str]:
    """Derive media type from the observed path and fetched bytes.

    Parameters
    ----------
    name:
        Broker-constructed or transport-observed final locator.
    body:
        Exact fetched bytes.

    Returns
    -------
    tuple[str, str]
        Authoritative media type and the derivation method.
    """

    suffix = Path(urllib.parse.urlsplit(name).path or name).suffix.lower()
    if suffix in _MEDIA_TYPES:
        return _MEDIA_TYPES[suffix], "path-extension"
    if body.startswith(b"%PDF-"):
        return "application/pdf", "content-sniff"
    try:
        json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    else:
        return "application/json", "content-sniff"
    try:
        body.decode("utf-8")
    except UnicodeDecodeError:
        return "application/octet-stream", "content-sniff"
    return "text/plain", "content-sniff"


def _failure_outcome(descriptor: Mapping[str, Any], outcome: str, *, detail: str) -> BrokerOutcome:
    """Build a typed failure outcome for one descriptor."""

    return BrokerOutcome(
        source_id=str(descriptor["source_id"]),
        kind=str(descriptor["kind"]),
        requested_role=str(descriptor["requested_role"]),
        bound_role=None,
        outcome=outcome,
        url=str(descriptor.get("url") or "") or None,
        final_url=None,
        redirect_chain=(),
        status=None,
        bytes_fetched=0,
        sha256=None,
        media_type=None,
        media_type_method=None,
        resolver_receipt=None,
        derived_citation=None,
        detail=detail,
    )


def _manifest_row(
    descriptor: Mapping[str, Any],
    outcome: BrokerOutcome,
) -> JsonObject:
    """Build a fresh manifest row solely from broker observations and receipts.

    Parameters
    ----------
    descriptor:
        Schema-shaped request retained only in explicitly named provenance fields.
    outcome:
        Successful broker result carrying transport and resolver observations.

    Returns
    -------
    dict[str, Any]
        Authoritative row with no spread or merge from the authored descriptor.
    """

    if outcome.sha256 is None or outcome.media_type is None or outcome.bound_role is None:
        raise SourceBrokerError("fetched broker outcome lacks machine-derived manifest facts")
    if outcome.resolver_receipt is not None:
        revision = outcome.resolver_receipt.get("resolved_sha")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise SourceBrokerError("forge manifest revision lacks a valid resolver receipt")
    else:
        revision = outcome.sha256
    kind = str(descriptor["kind"])
    authoritative_url = (
        outcome.url if kind == "forge-file" else outcome.final_url
    )
    if not authoritative_url or not outcome.final_url:
        raise SourceBrokerError("fetched broker outcome lacks an authoritative final url")
    row: JsonObject = {
        "source_id": str(descriptor["source_id"]),
        "url": authoritative_url,
        "final_url": outcome.final_url,
        "revision": revision,
        "expected_sha256": outcome.sha256,
        "media_type": outcome.media_type,
        "media_type_method": outcome.media_type_method,
        "broker_role": outcome.bound_role,
        "broker_outcome": OUTCOME_FETCHED,
        "redirect_chain": list(outcome.redirect_chain),
        "requested_role": str(descriptor["requested_role"]),
        "media_type_hint": str(descriptor.get("media_type_hint") or ""),
        "basis": str(descriptor["basis"]),
    }
    for name in ("repo", "path", "ref", "url", "identifier"):
        requested = str(descriptor.get(name) or "")
        if requested:
            row[f"requested_{name}"] = requested
    return row


def _classify_broker_role(
    descriptor: Mapping[str, Any],
    outcome: BrokerOutcome,
) -> str:
    """Classify one fetched object without copying the requested role.

    Parameters
    ----------
    descriptor:
        Authored request whose role is a non-authoritative intent.
    outcome:
        Machine-observed fetched object.

    Returns
    -------
    str
        Broker-owned role.
    """

    if descriptor["requested_role"] == "probe":
        return ROLE_PROBE
    locator = str(outcome.final_url or outcome.url or "")
    suffix = Path(urllib.parse.urlsplit(locator).path).suffix.lower()
    if suffix in _IMPLEMENTATION_SUFFIXES:
        return ROLE_IMPLEMENTATION
    return ROLE_DOCUMENTATION


# -- forge-file ------------------------------------------------------------


@dataclass(frozen=True)
class RefResolution:
    """One ref-resolution attempt: its SHA, its receipt, and whose failure it was.

    Parameters
    ----------
    sha:
        Machine-derived 40-hex commit SHA, or ``None`` when unresolved.
    receipt:
        The resolver receipt, recorded whether or not resolution succeeded.
    failure_outcome:
        The typed broker outcome for an unresolved ref. This is the field that
        keeps the record honest: only a forge answer that genuinely denies the
        ref earns :data:`OUTCOME_BAD_REF`.
    rate_limit:
        The throttling evidence, when the forge threw us out.
    """

    sha: Optional[str]
    receipt: JsonObject
    failure_outcome: Optional[str] = None
    rate_limit: Optional[RateLimitSignal] = None


def resolve_github_ref(
    repo: str,
    ref: str,
    *,
    transport: Transport,
    evidence_dir: Path,
    clock: Callable[[], str],
    timeout: float,
) -> RefResolution:
    """Resolve one confirmed ref to its immutable commit SHA, with a receipt.

    Failure is classified by *cause*, never collapsed. Only ``404``/``422`` --
    the forge stating authoritatively that the ref does not exist -- is a
    ``bad-ref``, because that is the only answer that is actually a claim about
    what the author supplied. Throttling is ``rate-limited``; a transport error,
    a server fault, or an unparseable response is ``unreachable``; a forge we
    cannot address at all is ``unsupported-forge``. Every one of those is our
    problem or the forge's, and saying otherwise in a durable record is a lie
    that also sends retries to repair a reference that was never broken.

    Parameters
    ----------
    repo:
        ``github.com/owner/name`` (scheme optional).
    ref:
        Tag, branch, or commit-ish the model confirmed exists.
    transport, evidence_dir, clock, timeout:
        Broker collaborators.

    Returns
    -------
    RefResolution
        The resolved SHA (or ``None``), the receipt, and the typed failure
        attribution. The SHA in any manifest row comes from this receipt,
        never the model.
    """

    matched = _GITHUB_REPO_PATTERN.match(repo)
    if matched is None:
        return RefResolution(
            sha=None,
            receipt={
                "receipt_kind": "ref-resolution",
                "forge": "unsupported",
                "repo": repo,
                "ref": ref,
                "resolved_sha": None,
                "resolved_at": clock(),
                "detail": "only github.com repositories are supported by the MVP resolver",
            },
            # A GitLab URL is a perfectly good reference we cannot yet
            # dereference. That is a gap in this resolver, not a bad ref.
            failure_outcome=OUTCOME_UNSUPPORTED_FORGE,
        )
    owner, name = matched.group(1), matched.group(2)
    endpoint = (
        "https://api.github.com/repos/"
        f"{urllib.parse.quote(owner)}/{urllib.parse.quote(name)}/commits/"
        f"{urllib.parse.quote(ref, safe='')}"
    )
    response = transport(endpoint, max_bytes=1024 * 1024, timeout=timeout)
    throttled = rate_limit_signal(response.status, response.headers)
    receipt: JsonObject = {
        "receipt_kind": "ref-resolution",
        "forge": "github",
        "repo": f"github.com/{owner}/{name}",
        "ref": ref,
        "endpoint": endpoint,
        "status": response.status,
        "response_sha256": hash_bytes(response.body) if response.body else None,
        "resolved_sha": None,
        "resolved_at": clock(),
        "detail": response.error or "",
        # Whether the client was authenticated is a fact about the run, not a
        # secret: it is the difference between a 60/hour and a 5,000/hour
        # ceiling, and a rung that throttles needs it visible. The credential
        # itself never leaves the request header.
        "credential_mode": _credential_mode(transport),
        "rate_limit": throttled.to_dict() if throttled is not None else None,
    }
    if throttled is not None:
        receipt["detail"] = (
            f"http {response.status}: forge rate limit reached "
            f"(remaining={throttled.remaining}, reset={throttled.reset_epoch}). "
            "The reference was not evaluated."
        )
        return RefResolution(
            sha=None,
            receipt=receipt,
            failure_outcome=OUTCOME_RATE_LIMITED,
            rate_limit=throttled,
        )
    if response.status == 200 and response.body:
        _store_evidence(evidence_dir, response.body)
        try:
            parsed = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            receipt["detail"] = "forge response was not JSON"
            return RefResolution(sha=None, receipt=receipt, failure_outcome=OUTCOME_UNREACHABLE)
        sha = parsed.get("sha") if isinstance(parsed, Mapping) else None
        if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha):
            receipt["resolved_sha"] = sha
            return RefResolution(sha=sha, receipt=receipt)
        receipt["detail"] = "forge response carried no commit sha"
        return RefResolution(sha=None, receipt=receipt, failure_outcome=OUTCOME_UNREACHABLE)
    if response.status in BAD_REF_STATUSES:
        if not receipt["detail"]:
            receipt["detail"] = f"http {response.status}: the forge has no such ref"
        return RefResolution(sha=None, receipt=receipt, failure_outcome=OUTCOME_BAD_REF)
    if not receipt["detail"]:
        receipt["detail"] = f"http {response.status}"
    return RefResolution(sha=None, receipt=receipt, failure_outcome=OUTCOME_UNREACHABLE)


def _credential_mode(transport: Transport) -> str:
    """Report *how* a transport authenticates to the forge, never *what* with.

    ``authenticated`` versus ``anonymous`` is the difference between a
    5,000/hour and a 60/hour ceiling, so a throttled rung needs it recorded --
    without it, "we were rate limited" cannot be told apart from "we were rate
    limited because nobody wired up a credential". The credential itself never
    leaves the request header.
    """

    reporter = getattr(transport, "github_credential_mode", None)
    if not callable(reporter):
        return "unknown"
    mode = str(reporter())
    return mode if mode in {"authenticated", "anonymous"} else "unknown"


def _broker_forge_file(
    descriptor: Mapping[str, Any],
    *,
    transport: Transport,
    evidence_dir: Path,
    clock: Callable[[], str],
    max_bytes: int,
    timeout: float,
) -> BrokerOutcome:
    """Resolve, pin, and fetch one repository file at an immutable SHA."""

    resolution = resolve_github_ref(
        str(descriptor["repo"]),
        str(descriptor["ref"]),
        transport=transport,
        evidence_dir=evidence_dir,
        clock=clock,
        timeout=timeout,
    )
    receipt = resolution.receipt
    if resolution.sha is None:
        failed = _failure_outcome(
            descriptor,
            resolution.failure_outcome or OUTCOME_UNREACHABLE,
            detail=str(receipt.get("detail", "")),
        )
        return replace(
            failed,
            status=receipt.get("status") if isinstance(receipt.get("status"), int) else None,
            resolver_receipt=receipt,
            rate_limit=(
                resolution.rate_limit.to_dict() if resolution.rate_limit is not None else None
            ),
        )
    sha = resolution.sha
    matched = _GITHUB_REPO_PATTERN.match(str(descriptor["repo"]))
    assert matched is not None  # resolve_github_ref already accepted it
    owner, name = matched.group(1), matched.group(2)
    raw_url = (
        f"https://raw.githubusercontent.com/{owner}/{name}/{sha}/"
        + urllib.parse.quote(str(descriptor["path"]))
    )
    fetched, body = _bounded_fetch(descriptor, raw_url, transport, max_bytes, timeout)
    if fetched.outcome != OUTCOME_FETCHED:
        return replace(fetched, resolver_receipt=receipt)
    _store_evidence(evidence_dir, body or b"")
    bound = _classify_broker_role(descriptor, fetched)
    return replace(fetched, bound_role=bound, resolver_receipt=receipt)


# -- raw-url and paper -----------------------------------------------------


def _bounded_fetch(
    descriptor: Mapping[str, Any],
    url: str,
    transport: Transport,
    max_bytes: int,
    timeout: float,
) -> tuple[BrokerOutcome, Optional[bytes]]:
    """Fetch one URL into a typed outcome, never raising for network failures.

    Returns
    -------
    tuple[BrokerOutcome, bytes | None]
        The typed outcome, plus the fetched body when the outcome is
        ``fetched`` so the caller can store it as evidence.
    """

    def make(
        outcome: str,
        *,
        final_url: Optional[str],
        redirect_chain: tuple[str, ...],
        status: Optional[int],
        bytes_fetched: int,
        sha256: Optional[str],
        detail: str,
    ) -> BrokerOutcome:
        return BrokerOutcome(
            source_id=str(descriptor["source_id"]),
            kind=str(descriptor["kind"]),
            requested_role=str(descriptor["requested_role"]),
            bound_role=None,
            outcome=outcome,
            url=url,
            final_url=final_url,
            redirect_chain=redirect_chain,
            status=status,
            bytes_fetched=bytes_fetched,
            sha256=sha256,
            media_type=None,
            media_type_method=None,
            resolver_receipt=None,
            derived_citation=None,
            detail=detail,
        )

    try:
        response = transport(url, max_bytes=max_bytes, timeout=timeout)
    except RedirectRefused as exc:
        return (
            make(
                OUTCOME_REDIRECT_REFUSED,
                final_url=exc.target,
                redirect_chain=exc.chain,
                status=None,
                bytes_fetched=0,
                sha256=None,
                detail=str(exc),
            ),
            None,
        )
    if response.truncated:
        return (
            make(
                OUTCOME_OVERSIZED,
                final_url=response.final_url,
                redirect_chain=response.redirect_chain,
                status=response.status,
                bytes_fetched=len(response.body),
                sha256=None,
                detail=f"body exceeded the {max_bytes}-byte ceiling",
            ),
            None,
        )
    throttled = rate_limit_signal(response.status, response.headers)
    if throttled is not None:
        # A throttled fetch is our budget running out, not a dead link. Typing it
        # as `unreachable` would send an author to look for a different source
        # when the one it named is fine and will be fetchable in an hour.
        return (
            replace(
                make(
                    OUTCOME_RATE_LIMITED,
                    final_url=response.final_url,
                    redirect_chain=response.redirect_chain,
                    status=response.status,
                    bytes_fetched=0,
                    sha256=None,
                    detail=(
                        f"http {response.status}: host rate limit reached "
                        f"(remaining={throttled.remaining}, reset={throttled.reset_epoch}). "
                        "The target was not evaluated."
                    ),
                ),
                rate_limit=throttled.to_dict(),
            ),
            None,
        )
    if response.status != 200 or response.error:
        return (
            make(
                OUTCOME_UNREACHABLE,
                final_url=response.final_url,
                redirect_chain=response.redirect_chain,
                status=response.status,
                bytes_fetched=0,
                sha256=None,
                detail=response.error or f"http {response.status}",
            ),
            None,
        )
    fetched = make(
            OUTCOME_FETCHED,
            final_url=response.final_url,
            redirect_chain=response.redirect_chain,
            status=response.status,
            bytes_fetched=len(response.body),
            sha256=hash_bytes(response.body),
            detail="",
        )
    media_type, method = _derive_media_type(response.final_url, response.body)
    return replace(fetched, media_type=media_type, media_type_method=method), response.body


def _broker_raw_url(
    descriptor: Mapping[str, Any],
    *,
    transport: Transport,
    evidence_dir: Path,
    clock: Callable[[], str],
    max_bytes: int,
    timeout: float,
) -> BrokerOutcome:
    """Fetch one direct URL target with a content-digest revision."""

    del clock
    url = str(descriptor["url"])
    if urllib.parse.urlsplit(url).scheme != "https":
        return _failure_outcome(
            descriptor, OUTCOME_UNREACHABLE, detail="only https urls are brokered"
        )
    fetched, body = _bounded_fetch(descriptor, url, transport, max_bytes, timeout)
    if fetched.outcome != OUTCOME_FETCHED:
        return fetched
    _store_evidence(evidence_dir, body or b"")
    bound = _classify_broker_role(descriptor, fetched)
    if bound == ROLE_PROBE:
        return replace(fetched, outcome=OUTCOME_PROBED, bound_role=ROLE_PROBE)
    return replace(fetched, bound_role=bound)


def _broker_paper(
    descriptor: Mapping[str, Any],
    *,
    transport: Transport,
    evidence_dir: Path,
    clock: Callable[[], str],
    max_bytes: int,
    timeout: float,
) -> BrokerOutcome:
    """Derive citation metadata from the authoritative registry for one paper.

    A target earns the ``introducing-paper`` role **only** when derivation
    succeeded — an author-declared role string can never relabel a code file
    as the paper. Papers do not enter the lane manifest in the MVP (their
    registry responses are not byte-stable across fetches); the derived
    citation and its raw registry response are the machine half of the
    citation contract.
    """

    reference = str(descriptor.get("url") or descriptor.get("identifier") or "")
    citation, receipt = derive_paper_metadata(
        reference,
        transport=transport,
        evidence_dir=evidence_dir,
        clock=clock,
        max_bytes=max_bytes,
        timeout=timeout,
    )
    if citation is None:
        failed = _failure_outcome(
            descriptor,
            OUTCOME_UNREACHABLE,
            detail=str(receipt.get("detail", "citation derivation failed")),
        )
        return replace(failed, resolver_receipt=receipt)
    citation = {**citation, "source_id": str(descriptor["source_id"])}
    return BrokerOutcome(
        source_id=str(descriptor["source_id"]),
        kind="paper",
        requested_role=str(descriptor["requested_role"]),
        bound_role=ROLE_INTRODUCING_PAPER,
        outcome=OUTCOME_PAPER_DERIVATION_ONLY,
        url=reference or None,
        final_url=str(receipt.get("endpoint") or "") or None,
        redirect_chain=(),
        status=int(receipt["status"]) if receipt.get("status") is not None else None,
        bytes_fetched=int(receipt.get("bytes", 0)),
        sha256=receipt.get("response_sha256"),
        media_type=None,
        media_type_method=None,
        resolver_receipt=receipt,
        derived_citation=citation,
        detail="",
    )


def derive_paper_metadata(
    reference: str,
    *,
    transport: Transport,
    evidence_dir: Path,
    clock: Callable[[], str],
    max_bytes: int = 4 * 1024 * 1024,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[Optional[JsonObject], JsonObject]:
    """Derive a citation record from an arXiv ID, DOI, or OpenReview ID.

    Parameters
    ----------
    reference:
        URL or bare identifier carrying (or resolving to) a paper identity.
    transport, evidence_dir, clock, max_bytes, timeout:
        Broker collaborators and budgets.

    Returns
    -------
    tuple[dict | None, dict]
        The ``derived_citation`` record (or ``None``) and the derivation
        receipt including the raw registry response digest.
    """

    arxiv = _ARXIV_ID_PATTERN.search(reference)
    if arxiv is not None:
        return _derive_arxiv(
            arxiv.group(1), transport, evidence_dir, clock, max_bytes, timeout
        )
    openreview = _OPENREVIEW_PATTERN.search(reference)
    if openreview is not None:
        return _derive_openreview(
            openreview.group(1), transport, evidence_dir, clock, max_bytes, timeout
        )
    doi = _DOI_PATTERN.search(reference + " ")
    if doi is not None:
        return _derive_crossref(
            doi.group(1), transport, evidence_dir, clock, max_bytes, timeout
        )
    return None, {
        "receipt_kind": "citation-derivation",
        "reference": reference,
        "registry": None,
        "status": None,
        "detail": "no resolvable arXiv, DOI, or OpenReview identifier in the reference",
        "resolved_at": clock(),
    }


def _registry_receipt(
    registry: str,
    identifier: str,
    endpoint: str,
    response: TransportResponse,
    clock: Callable[[], str],
) -> JsonObject:
    """Build the shared derivation receipt shell."""

    return {
        "receipt_kind": "citation-derivation",
        "registry": registry,
        "identifier": identifier,
        "endpoint": endpoint,
        "status": response.status,
        "bytes": len(response.body),
        "response_sha256": hash_bytes(response.body) if response.body else None,
        "detail": response.error or "",
        "resolved_at": clock(),
    }


def _derive_arxiv(
    arxiv_id: str,
    transport: Transport,
    evidence_dir: Path,
    clock: Callable[[], str],
    max_bytes: int,
    timeout: float,
) -> tuple[Optional[JsonObject], JsonObject]:
    """Derive a citation from the arXiv Atom API."""

    endpoint = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    response = transport(endpoint, max_bytes=max_bytes, timeout=timeout)
    receipt = _registry_receipt("arxiv", arxiv_id, endpoint, response, clock)
    if response.status != 200 or not response.body:
        return None, receipt
    _store_evidence(evidence_dir, response.body)
    try:
        root = ElementTree.fromstring(response.body.decode("utf-8"))
    except (UnicodeDecodeError, ElementTree.ParseError) as exc:
        receipt["detail"] = f"arXiv response unparseable: {exc}"
        return None, receipt
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None:
        receipt["detail"] = "arXiv response carried no entry"
        return None, receipt
    title = _text(entry.find("atom:title", ns))
    authors = [
        _text(author.find("atom:name", ns))
        for author in entry.findall("atom:author", ns)
        if _text(author.find("atom:name", ns))
    ]
    published = _text(entry.find("atom:published", ns))
    if not title or not authors:
        receipt["detail"] = "arXiv entry lacked title or authors"
        return None, receipt
    return (
        {
            "citation_kind": "derived_citation",
            "registry": "arxiv",
            "title": title,
            "authors": authors,
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "venue": "arXiv",
            "identifiers": {"arxiv": arxiv_id},
            "response_sha256": receipt["response_sha256"],
        },
        receipt,
    )


def _derive_crossref(
    doi: str,
    transport: Transport,
    evidence_dir: Path,
    clock: Callable[[], str],
    max_bytes: int,
    timeout: float,
) -> tuple[Optional[JsonObject], JsonObject]:
    """Derive a citation from the Crossref works API."""

    endpoint = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/')}"
    response = transport(endpoint, max_bytes=max_bytes, timeout=timeout)
    receipt = _registry_receipt("crossref", doi, endpoint, response, clock)
    if response.status != 200 or not response.body:
        return None, receipt
    _store_evidence(evidence_dir, response.body)
    try:
        message = json.loads(response.body.decode("utf-8")).get("message", {})
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        receipt["detail"] = f"Crossref response unparseable: {exc}"
        return None, receipt
    titles = message.get("title") or []
    authors = [
        " ".join(part for part in (author.get("given"), author.get("family")) if part)
        for author in message.get("author", [])
        if isinstance(author, Mapping)
    ]
    issued = message.get("issued", {}).get("date-parts", [[None]])
    year = issued[0][0] if issued and issued[0] else None
    container = message.get("container-title") or []
    if not titles or not authors:
        receipt["detail"] = "Crossref message lacked title or authors"
        return None, receipt
    return (
        {
            "citation_kind": "derived_citation",
            "registry": "crossref",
            "title": str(titles[0]),
            "authors": [author for author in authors if author],
            "year": int(year) if isinstance(year, int) else None,
            "venue": str(container[0]) if container else None,
            "identifiers": {"doi": doi},
            "response_sha256": receipt["response_sha256"],
        },
        receipt,
    )


def _derive_openreview(
    note_id: str,
    transport: Transport,
    evidence_dir: Path,
    clock: Callable[[], str],
    max_bytes: int,
    timeout: float,
) -> tuple[Optional[JsonObject], JsonObject]:
    """Derive a citation from the OpenReview notes API."""

    endpoint = f"https://api.openreview.net/notes?id={urllib.parse.quote(note_id)}"
    response = transport(endpoint, max_bytes=max_bytes, timeout=timeout)
    receipt = _registry_receipt("openreview", note_id, endpoint, response, clock)
    if response.status != 200 or not response.body:
        return None, receipt
    _store_evidence(evidence_dir, response.body)
    try:
        notes = json.loads(response.body.decode("utf-8")).get("notes", [])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        receipt["detail"] = f"OpenReview response unparseable: {exc}"
        return None, receipt
    if not notes:
        receipt["detail"] = "OpenReview returned no notes"
        return None, receipt
    content = notes[0].get("content", {}) if isinstance(notes[0], Mapping) else {}

    def _value(name: str) -> Any:
        value = content.get(name)
        if isinstance(value, Mapping):
            return value.get("value")
        return value

    title = _value("title")
    authors = _value("authors")
    year = _value("year")
    if not title or not isinstance(authors, list) or not authors:
        receipt["detail"] = "OpenReview note lacked title or authors"
        return None, receipt
    return (
        {
            "citation_kind": "derived_citation",
            "registry": "openreview",
            "title": str(title),
            "authors": [str(author) for author in authors],
            "year": int(year) if isinstance(year, (int, str)) and str(year).isdigit() else None,
            "venue": None,
            "identifiers": {"openreview": note_id},
            "response_sha256": receipt["response_sha256"],
        },
        receipt,
    )


def _text(node: Optional[ElementTree.Element]) -> str:
    """Return one element's normalized text content."""

    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split())


def _store_evidence(evidence_dir: Path, body: bytes) -> Optional[Path]:
    """Store raw evidence bytes content-addressed under the broker directory."""

    if not body:
        return None
    evidence_dir.mkdir(parents=True, exist_ok=True)
    digest = hash_bytes(body).removeprefix("sha256:")
    destination = evidence_dir / f"{digest[:16]}.bin"
    if not destination.exists():
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(body)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination
