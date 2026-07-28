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
(``fetched | oversized | unreachable | redirect-refused | bad-ref``): one dead
link never aborts the pack, and every failure carries its receipt so stage 2
and the validators see exactly what exists.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Union

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
OUTCOME_PAPER_DERIVATION_ONLY = "paper-derivation-only"
OUTCOME_PROBED = "probed"

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


class SourceBrokerError(ValueError):
    """Raised when a discovery descriptor cannot be brokered at all."""


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
    """

    status: int
    final_url: str
    redirect_chain: tuple[str, ...]
    body: bytes
    truncated: bool
    error: Optional[str] = None


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
    """Streaming HTTPS transport with per-hop redirect policy and byte ceiling."""

    def __init__(
        self,
        *,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        redirect_allowlist: frozenset[str] = REDIRECT_HOST_ALLOWLIST,
        user_agent: str = "menagerie-crawler-source-broker/1",
    ) -> None:
        self.max_redirects = max_redirects
        self.redirect_allowlist = redirect_allowlist
        self.user_agent = user_agent
        self._opener = urllib.request.build_opener(_NoRedirect())

    def __call__(self, url: str, *, max_bytes: int, timeout: float) -> TransportResponse:
        """Fetch one URL, following only allowlisted redirects, streaming-bounded."""

        chain: list[str] = [url]
        current = url
        for _hop in range(self.max_redirects + 1):
            request = urllib.request.Request(
                current,
                headers={"User-Agent": self.user_agent, "Accept": "*/*"},
            )
            try:
                with self._opener.open(request, timeout=timeout) as response:
                    body, truncated = _read_bounded(response, max_bytes)
                    return TransportResponse(
                        status=int(getattr(response, "status", 200) or 200),
                        final_url=current,
                        redirect_chain=tuple(chain),
                        body=body,
                        truncated=truncated,
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
    "truncated", "error"}``. An unmapped URL is unreachable — the fixture
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
        return TransportResponse(
            status=int(entry.get("status", 200)),
            final_url=str(entry.get("final_url", url)),
            redirect_chain=chain,
            body=body,
            truncated=truncated,
            error=(str(entry["error"]) if entry.get("error") else None),
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
    resolver_receipt: Optional[JsonObject]
    derived_citation: Optional[JsonObject]
    detail: str = ""

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
            "resolver_receipt": self.resolver_receipt,
            "derived_citation": self.derived_citation,
            "detail": self.detail,
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
    descriptors: list[Mapping[str, Any]],
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
    for position, raw in enumerate(descriptors):
        descriptor = _validated_descriptor(raw, position)
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
            revision = (
                outcome.resolver_receipt["resolved_sha"]
                if outcome.resolver_receipt is not None
                else f"sha256-{str(outcome.sha256).removeprefix('sha256:')[:12]}"
            )
            pack.rows.append(
                {
                    "source_id": descriptor["source_id"],
                    "url": outcome.url,
                    "revision": revision,
                    "expected_sha256": outcome.sha256,
                    "media_type": descriptor["media_type"],
                    "broker_role": outcome.bound_role,
                    "broker_outcome": OUTCOME_FETCHED,
                }
            )
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


# -- descriptor validation -------------------------------------------------


_DESCRIPTOR_KINDS = frozenset({"forge-file", "raw-url", "paper"})
_REQUESTED_ROLES = frozenset({"implementation", "paper", "documentation", "probe"})


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
    role = str(raw.get("role", "implementation"))
    if role not in _REQUESTED_ROLES:
        raise SourceBrokerError(f"source descriptor {position} role {role!r} is not supported")
    source_id = str(raw.get("source_id", "")).strip()
    if not source_id:
        raise SourceBrokerError(f"source descriptor {position} must carry a source_id")
    for forbidden in ("revision", "expected_sha256", "sha256", "commit_sha"):
        if raw.get(forbidden):
            raise SourceBrokerError(
                f"source descriptor {position} carries machine-owned field {forbidden!r}; "
                "exact strings are derived by the broker, never authored"
            )
    descriptor: JsonObject = {
        "source_id": source_id,
        "kind": kind,
        "role": role,
        "repo": str(raw.get("repo", "")).strip(),
        "path": str(raw.get("path", "")).strip(),
        "ref": str(raw.get("ref", "")).strip(),
        "url": str(raw.get("url", "")).strip(),
        "identifier": str(raw.get("identifier", "")).strip(),
        "media_type": str(raw.get("media_type", "")).strip(),
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
    if not descriptor["media_type"]:
        descriptor["media_type"] = _guess_media_type(descriptor["path"] or descriptor["url"])
    return descriptor


def _guess_media_type(name: str) -> str:
    """Return the media type for a path or URL by extension."""

    suffix = Path(urllib.parse.urlsplit(name).path or name).suffix.lower()
    return _MEDIA_TYPES.get(suffix, "application/octet-stream")


def _failure_outcome(descriptor: Mapping[str, Any], outcome: str, *, detail: str) -> BrokerOutcome:
    """Build a typed failure outcome for one descriptor."""

    return BrokerOutcome(
        source_id=str(descriptor["source_id"]),
        kind=str(descriptor["kind"]),
        requested_role=str(descriptor["role"]),
        bound_role=None,
        outcome=outcome,
        url=str(descriptor.get("url") or "") or None,
        final_url=None,
        redirect_chain=(),
        status=None,
        bytes_fetched=0,
        sha256=None,
        resolver_receipt=None,
        derived_citation=None,
        detail=detail,
    )


# -- forge-file ------------------------------------------------------------


def resolve_github_ref(
    repo: str,
    ref: str,
    *,
    transport: Transport,
    evidence_dir: Path,
    clock: Callable[[], str],
    timeout: float,
) -> tuple[Optional[str], JsonObject]:
    """Resolve one confirmed ref to its immutable commit SHA, with a receipt.

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
    tuple[str | None, dict[str, Any]]
        The resolved 40-hex SHA (or ``None``) and the resolver receipt. The
        SHA in any manifest row comes from this receipt, never the model.
    """

    matched = _GITHUB_REPO_PATTERN.match(repo)
    if matched is None:
        return None, {
            "receipt_kind": "ref-resolution",
            "forge": "unsupported",
            "repo": repo,
            "ref": ref,
            "resolved_sha": None,
            "resolved_at": clock(),
            "detail": "only github.com repositories are supported by the MVP resolver",
        }
    owner, name = matched.group(1), matched.group(2)
    endpoint = (
        "https://api.github.com/repos/"
        f"{urllib.parse.quote(owner)}/{urllib.parse.quote(name)}/commits/"
        f"{urllib.parse.quote(ref, safe='')}"
    )
    response = transport(endpoint, max_bytes=1024 * 1024, timeout=timeout)
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
    }
    if response.status == 200 and response.body:
        _store_evidence(evidence_dir, response.body)
        try:
            parsed = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            receipt["detail"] = "forge response was not JSON"
            return None, receipt
        sha = parsed.get("sha") if isinstance(parsed, Mapping) else None
        if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha):
            receipt["resolved_sha"] = sha
            return sha, receipt
        receipt["detail"] = "forge response carried no commit sha"
    return None, receipt


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

    sha, receipt = resolve_github_ref(
        str(descriptor["repo"]),
        str(descriptor["ref"]),
        transport=transport,
        evidence_dir=evidence_dir,
        clock=clock,
        timeout=timeout,
    )
    if sha is None:
        failed = _failure_outcome(
            descriptor, OUTCOME_BAD_REF, detail=str(receipt.get("detail", ""))
        )
        return replace(failed, resolver_receipt=receipt)
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
    role = str(descriptor["role"])
    bound = ROLE_IMPLEMENTATION if role == "implementation" else ROLE_DOCUMENTATION
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

    common = {
        "source_id": str(descriptor["source_id"]),
        "kind": str(descriptor["kind"]),
        "requested_role": str(descriptor["role"]),
        "bound_role": None,
        "resolver_receipt": None,
        "derived_citation": None,
    }
    try:
        response = transport(url, max_bytes=max_bytes, timeout=timeout)
    except RedirectRefused as exc:
        return (
            BrokerOutcome(
                **common,
                outcome=OUTCOME_REDIRECT_REFUSED,
                url=url,
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
            BrokerOutcome(
                **common,
                outcome=OUTCOME_OVERSIZED,
                url=url,
                final_url=response.final_url,
                redirect_chain=response.redirect_chain,
                status=response.status,
                bytes_fetched=len(response.body),
                sha256=None,
                detail=f"body exceeded the {max_bytes}-byte ceiling",
            ),
            None,
        )
    if response.status != 200 or response.error:
        return (
            BrokerOutcome(
                **common,
                outcome=OUTCOME_UNREACHABLE,
                url=url,
                final_url=response.final_url,
                redirect_chain=response.redirect_chain,
                status=response.status,
                bytes_fetched=0,
                sha256=None,
                detail=response.error or f"http {response.status}",
            ),
            None,
        )
    return (
        BrokerOutcome(
            **common,
            outcome=OUTCOME_FETCHED,
            url=url,
            final_url=response.final_url,
            redirect_chain=response.redirect_chain,
            status=response.status,
            bytes_fetched=len(response.body),
            sha256=hash_bytes(response.body),
            detail="",
        ),
        response.body,
    )


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
    role = str(descriptor["role"])
    if role == "probe":
        return replace(fetched, outcome=OUTCOME_PROBED, bound_role=ROLE_PROBE)
    bound = ROLE_IMPLEMENTATION if role == "implementation" else ROLE_DOCUMENTATION
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
        requested_role=str(descriptor["role"]),
        bound_role=ROLE_INTRODUCING_PAPER,
        outcome=OUTCOME_PAPER_DERIVATION_ONLY,
        url=reference or None,
        final_url=str(receipt.get("endpoint") or "") or None,
        redirect_chain=(),
        status=int(receipt["status"]) if receipt.get("status") is not None else None,
        bytes_fetched=int(receipt.get("bytes", 0)),
        sha256=receipt.get("response_sha256"),
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
