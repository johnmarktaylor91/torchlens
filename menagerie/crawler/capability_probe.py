"""Nonce-bound proof that the author path really has live web research tools.

The doctor's ``author-web-tools`` check used to grep ``claude --help`` for tool-name
strings. That could never pass, and worse, it proved nothing: a name in a help text is not
a working tool. This module replaces it with a probe that the author path can only satisfy
by *actually using* the tools.

How the proof works
-------------------
The doctor mints a fresh nonce and sends a ``author-capability-probe.v1`` request down the
real author path (operator wrapper -> author queue -> a real author subagent). The pool
derives a **challenge from that nonce**: an index into a fixed corpus of PyPI projects
selects one package, and the session must report that package's *current* released version
and release serial, gathered independently through each required tool.

Four properties make it a proof rather than an assertion:

1. **Unpredictable target.** Which package is demanded is a function of a nonce minted
   seconds earlier, so no answer can be precomputed, cached, or hardcoded. An auditor
   holding the nonce can recompute the demanded package and re-check the answer.
2. **Live-only answer.** The demanded facts (latest version, release serial) change over
   time and are not derivable offline. An author path with no web reach cannot produce
   them at all -- which is exactly the capability the doctor is gating on.
3. **Cross-tool corroboration.** All three tools must independently report the *same*
   version. Losing any one tool loses one of the three independent reports, and the
   probe fails on the missing report rather than on a claim about tool availability.
4. **Tool-shaped evidence.** Each tool must return the evidence only it produces: a search
   tool returns multiple ranked results from more than one host; the Exa fetch tool
   returns a long verbatim document body, self-consistent with its own declared digest and
   literally containing the reported version. Two search tools that returned byte-identical
   result lists are refused, so one tool's output cannot be pasted into another's slot.

Threat model, stated honestly
-----------------------------
This proves the author path has working live web grounding, and corroborates each tool
against an unpredictable live fact. It does **not** defend against an author subagent that
deliberately fabricates a self-consistent document -- that agent is inside our own trust
boundary and is the thing being configured, not an adversary. What it does defend against,
and what the old check could not detect at all, is the real failure mode: a pool whose
tools are missing, unconfigured, MCP-disconnected, or silently erroring.

Nothing here may be relaxed to make a campaign start. A failing probe means the author
path cannot research, and a campaign started in that state produces ungrounded proposals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import re
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlsplit

from menagerie.crawler.identity import stable_hash, utc_now
from menagerie.crawler.models import JsonObject

CAPABILITY_PROBE_FORMAT = "menagerie.crawler.author-capability-probe.v1"
CAPABILITY_RECEIPT_FORMAT = "menagerie.crawler.author-capability-receipt.v1"

#: Tools the author path must genuinely have. The doctor requires a receipt for each.
#: These are *canonical* names. See :data:`TOOL_NAME_RESOLUTION`.
REQUIRED_AUTHOR_TOOLS = ("WebSearch", "web_search_exa", "web_fetch_exa")

#: How a canonical name relates to the name a tool is actually *registered* under.
#:
#: The Exa tools reach a session through MCP, and the registered name carries a namespace
#: prefix that depends on how the session was launched: a plugin-loaded server registers
#: ``mcp__plugin_everything-claude-code_exa__web_search_exa`` while an explicit
#: ``--mcp-config`` naming the server ``exa`` registers ``mcp__exa__web_search_exa``. Only
#: the suffix is stable across those contexts, so the suffix -- not any single literal --
#: is the identity. Both ends of the probe resolve by suffix and record the canonical name,
#: which is why a namespace change can never read as a missing tool.
TOOL_NAME_RESOLUTION = "canonical-name-is-the-stable-suffix-of-the-registered-name"

#: Separators a namespace prefix may use before the canonical suffix. Requiring one of
#: these keeps resolution from accepting an unrelated tool that merely ends in the same
#: letters (``my_web_search_exa`` is not ``web_search_exa``).
_TOOL_NAMESPACE_SEPARATORS = ("__", ".", ":", "/")

#: Fixed challenge corpus. Every entry is a long-lived, actively maintained PyPI project
#: with a stable JSON metadata endpoint. Changing this list changes ``CORPUS_REVISION``,
#: which is recorded in every receipt so an auditor can reproduce the selection.
CHALLENGE_CORPUS = (
    "requests",
    "numpy",
    "pandas",
    "scipy",
    "flask",
    "django",
    "sqlalchemy",
    "pydantic",
    "click",
    "rich",
    "attrs",
    "jinja2",
    "pytest",
    "coverage",
    "mypy",
    "ruff",
    "packaging",
    "setuptools",
    "wheel",
    "urllib3",
    "certifi",
    "pyyaml",
    "jsonschema",
    "typing-extensions",
    "platformdirs",
    "filelock",
    "tqdm",
    "networkx",
    "pillow",
    "matplotlib",
    "beautifulsoup4",
    "lxml",
)
CORPUS_REVISION = "capability-corpus-v1"

#: Minimum verbatim document length a fetch tool must return. A search snippet is far
#: shorter than a real metadata document, so this separates fetch evidence from search
#: evidence structurally.
MIN_FETCH_CONTENT_CHARS = 800

#: Minimum long-form excerpt length the Exa search tool must return for at least one
#: result. Exa returns page text, not just a title and a one-line snippet.
MIN_EXA_EXCERPT_CHARS = 200

#: Conservative floor for a PyPI release serial. A fabricated placeholder such as ``0``
#: or ``1`` is refused; the real values are orders of magnitude larger.
MIN_RELEASE_SERIAL = 1_000_000

_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)*(?:[.\-+_a-zA-Z0-9]*)$")
_FORBIDDEN_HOST_SUFFIXES = (".invalid", ".example", ".test", ".localhost")


class CapabilityProbeError(ValueError):
    """Raised when capability evidence fails to prove the tool was exercised."""


@dataclass(frozen=True)
class CapabilityChallenge:
    """One nonce-derived research challenge.

    Parameters
    ----------
    nonce:
        Doctor-minted nonce the whole probe is bound to.
    package:
        Corpus package selected by the nonce.
    metadata_url:
        Exact machine-readable document the fetch tool must retrieve.
    project_url:
        Human-readable project page the search tools should surface.
    challenge_id:
        Stable digest over nonce, package, and corpus revision.
    """

    nonce: str
    package: str
    metadata_url: str
    project_url: str
    challenge_id: str

    def to_dict(self) -> JsonObject:
        """Return the JSON challenge published to the author session.

        Returns
        -------
        dict[str, Any]
            Closed challenge payload.
        """

        return {
            "challenge_id": self.challenge_id,
            "corpus_revision": CORPUS_REVISION,
            "package": self.package,
            "metadata_url": self.metadata_url,
            "project_url": self.project_url,
            "required_answers": ["version", "last_serial"],
        }


def derive_challenge(nonce: str) -> CapabilityChallenge:
    """Select the challenge this nonce demands.

    Parameters
    ----------
    nonce:
        Doctor-minted probe nonce.

    Returns
    -------
    CapabilityChallenge
        Deterministically selected challenge. Deterministic selection is what lets an
        auditor recompute which package was demanded; unpredictability comes from the
        nonce, not from hiding the rule.

    Raises
    ------
    CapabilityProbeError
        When the nonce is empty.
    """

    if not nonce or not str(nonce).strip():
        raise CapabilityProbeError("capability probe requires a nonempty nonce")
    digest = hashlib.sha256(f"{CORPUS_REVISION}:{nonce}".encode("utf-8")).hexdigest()
    package = CHALLENGE_CORPUS[int(digest[:16], 16) % len(CHALLENGE_CORPUS)]
    return CapabilityChallenge(
        nonce=str(nonce),
        package=package,
        metadata_url=f"https://pypi.org/pypi/{package}/json",
        project_url=f"https://pypi.org/project/{package}/",
        challenge_id=stable_hash(
            {"nonce": str(nonce), "package": package, "corpus": CORPUS_REVISION}
        ),
    )


def canonical_tool_name(observed: Any) -> Optional[str]:
    """Resolve one registered tool name to the canonical tool it provides.

    Parameters
    ----------
    observed:
        Name a tool is registered under in some execution context, which may carry a
        namespace prefix (``mcp__exa__web_search_exa``) or be the canonical name itself.

    Returns
    -------
    str | None
        Canonical required-tool name, or ``None`` when the name names no required tool.
        Resolution is spelling-only: it decides *which* capability an entry claims, never
        whether the claim is true. Every evidence check still has to pass afterwards.
    """

    name = str(observed or "").strip()
    if not name:
        return None
    if name in REQUIRED_AUTHOR_TOOLS:
        return name
    for canonical in REQUIRED_AUTHOR_TOOLS:
        if not name.endswith(canonical) or len(name) == len(canonical):
            continue
        prefix = name[: -len(canonical)]
        if prefix.endswith(_TOOL_NAMESPACE_SEPARATORS):
            return canonical
    return None


def _canonicalize_tool_evidence(
    tools: Mapping[str, Any]
) -> dict[str, tuple[str, Mapping[str, Any]]]:
    """Key per-tool evidence by canonical name, retaining the registered spelling.

    Parameters
    ----------
    tools:
        Session-reported evidence keyed by whatever name the session called the tool by.

    Returns
    -------
    dict[str, tuple[str, Mapping[str, Any]]]
        Canonical tool name to ``(registered_name, evidence)``.

    Raises
    ------
    CapabilityProbeError
        When two keys claim the same canonical tool, or an entry's declared
        ``registered_tool_name`` resolves to a different tool than its key. Both are
        refusals: an ambiguous claim is never silently reduced to one of its readings.
    """

    resolved: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for raw_name, entry in tools.items():
        canonical = canonical_tool_name(raw_name)
        if canonical is None:
            continue
        if not isinstance(entry, Mapping):
            raise CapabilityProbeError(f"{raw_name} evidence is not an object")
        registered = str(entry.get("registered_tool_name") or raw_name).strip()
        if canonical_tool_name(registered) != canonical:
            raise CapabilityProbeError(
                f"{raw_name} evidence declares registered_tool_name {registered!r}, which is "
                "not a spelling of the same tool"
            )
        if canonical in resolved:
            raise CapabilityProbeError(
                f"capability evidence claims {canonical} twice, as "
                f"{resolved[canonical][0]!r} and {registered!r}"
            )
        resolved[canonical] = (registered, entry)
    return resolved


def build_probe_request(
    *,
    nonce: str,
    required_output_path: str,
    requested_at: Optional[str] = None,
    deadline_seconds: int = 120,
) -> JsonObject:
    """Build the doctor-shaped capability request.

    Parameters
    ----------
    nonce:
        Fresh probe nonce.
    required_output_path:
        Exact absolute path the receipt must be published to.
    requested_at:
        Request instant; defaults to now.
    deadline_seconds:
        Freshness window.

    Returns
    -------
    dict[str, Any]
        Request payload matching the doctor's envelope.
    """

    return {
        "format": CAPABILITY_PROBE_FORMAT,
        "nonce": nonce,
        "requested_at": requested_at or utc_now(),
        "deadline_seconds": int(deadline_seconds),
        "required_tools": list(REQUIRED_AUTHOR_TOOLS),
        "tool_name_resolution": TOOL_NAME_RESOLUTION,
        "required_output_path": required_output_path,
    }


def _parse_instant(value: Any, label: str) -> datetime:
    """Parse one ISO-8601 instant into an aware datetime.

    Parameters
    ----------
    value:
        Raw timestamp.
    label:
        Field name for diagnostics.

    Returns
    -------
    datetime
        Timezone-aware instant.

    Raises
    ------
    CapabilityProbeError
        When the timestamp is absent or unparseable.
    """

    try:
        text = str(value).strip()
        parsed = datetime.fromisoformat(text.removesuffix("Z") + "+00:00" if text.endswith("Z") else text)
    except (TypeError, ValueError) as exc:
        raise CapabilityProbeError(f"capability evidence has an unparseable {label}: {exc}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _require_live_url(raw: Any, label: str) -> str:
    """Validate one absolute, non-placeholder web URL.

    Parameters
    ----------
    raw:
        Candidate URL.
    label:
        Field name for diagnostics.

    Returns
    -------
    str
        Normalized URL.

    Raises
    ------
    CapabilityProbeError
        When the URL is relative, non-HTTP, or points at a reserved placeholder host.
    """

    url = str(raw or "").strip()
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise CapabilityProbeError(f"capability evidence {label} is not an absolute web URL: {url!r}")
    host = parts.netloc.split("@")[-1].split(":")[0].lower()
    if any(host.endswith(suffix) for suffix in _FORBIDDEN_HOST_SUFFIXES):
        raise CapabilityProbeError(f"capability evidence {label} names a placeholder host: {host}")
    return url


def _require_version(raw: Any, label: str) -> str:
    """Validate one reported release version string.

    Parameters
    ----------
    raw:
        Candidate version.
    label:
        Field name for diagnostics.

    Returns
    -------
    str
        Normalized version.

    Raises
    ------
    CapabilityProbeError
        When the version is absent or not version-shaped.
    """

    version = str(raw or "").strip()
    if not version or len(version) > 32 or not _VERSION_PATTERN.match(version):
        raise CapabilityProbeError(f"capability evidence {label} is not a version string: {raw!r}")
    return version


def _search_results(entry: Mapping[str, Any], tool: str, minimum: int) -> list[tuple[str, str]]:
    """Validate one search tool's ranked results.

    Parameters
    ----------
    entry:
        Per-tool evidence object.
    tool:
        Tool name for diagnostics.
    minimum:
        Minimum number of distinct results a real search must return.

    Returns
    -------
    list[tuple[str, str]]
        ``(url, excerpt)`` pairs in reported order.

    Raises
    ------
    CapabilityProbeError
        When the results are missing, too few, or not distinct.
    """

    raw = entry.get("results")
    if not isinstance(raw, list) or len(raw) < minimum:
        raise CapabilityProbeError(
            f"{tool} evidence must carry at least {minimum} ranked results"
        )
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise CapabilityProbeError(f"{tool} result {index} is not an object")
        url = _require_live_url(item.get("url"), f"{tool}.results[{index}].url")
        title = str(item.get("title") or "").strip()
        if not title:
            raise CapabilityProbeError(f"{tool} result {index} carries no title")
        body = max(
            str(item.get("excerpt") or ""), str(item.get("text") or ""), key=len
        )
        pairs.append((url, body))
    if len({url for url, _ in pairs}) != len(pairs):
        raise CapabilityProbeError(f"{tool} evidence repeats the same result URL")
    return pairs


def _validate_websearch(entry: Mapping[str, Any], challenge: CapabilityChallenge) -> str:
    """Validate ``WebSearch`` evidence and return its reported version.

    Parameters
    ----------
    entry:
        Per-tool evidence object.
    challenge:
        Active challenge.

    Returns
    -------
    str
        Version this tool independently reported.

    Raises
    ------
    CapabilityProbeError
        When the evidence is not what a live web search returns.
    """

    pairs = _search_results(entry, "WebSearch", minimum=2)
    hosts = {urlsplit(url).netloc.lower() for url, _ in pairs}
    if len(hosts) < 2:
        raise CapabilityProbeError(
            "WebSearch evidence covers a single host; a live search ranks across sources"
        )
    if not any(challenge.package.lower() in url.lower() for url, _ in pairs):
        raise CapabilityProbeError(
            f"WebSearch evidence names no result for the demanded package {challenge.package}"
        )
    return _require_version(entry.get("reported_version"), "WebSearch.reported_version")


def _validate_exa_search(entry: Mapping[str, Any], challenge: CapabilityChallenge) -> str:
    """Validate ``web_search_exa`` evidence and return its reported version.

    Parameters
    ----------
    entry:
        Per-tool evidence object.
    challenge:
        Active challenge.

    Returns
    -------
    str
        Version this tool independently reported.

    Raises
    ------
    CapabilityProbeError
        When the evidence lacks the long-form page text Exa returns.
    """

    pairs = _search_results(entry, "web_search_exa", minimum=2)
    if not any(len(excerpt.strip()) >= MIN_EXA_EXCERPT_CHARS for _, excerpt in pairs):
        raise CapabilityProbeError(
            "web_search_exa evidence carries no long-form page text; Exa returns document "
            f"content of at least {MIN_EXA_EXCERPT_CHARS} characters"
        )
    if not any(challenge.package.lower() in url.lower() for url, _ in pairs):
        raise CapabilityProbeError(
            f"web_search_exa evidence names no result for the demanded package "
            f"{challenge.package}"
        )
    return _require_version(entry.get("reported_version"), "web_search_exa.reported_version")


def _validate_exa_fetch(entry: Mapping[str, Any], challenge: CapabilityChallenge) -> str:
    """Validate ``web_fetch_exa`` evidence and return its reported version.

    Parameters
    ----------
    entry:
        Per-tool evidence object.
    challenge:
        Active challenge.

    Returns
    -------
    str
        Version this tool independently reported.

    Raises
    ------
    CapabilityProbeError
        When the fetched document is the wrong URL, too short to be a real document,
        inconsistent with its own declared digest, or does not contain the reported facts.
    """

    url = _require_live_url(entry.get("url"), "web_fetch_exa.url")
    if url.rstrip("/") != challenge.metadata_url.rstrip("/"):
        raise CapabilityProbeError(
            f"web_fetch_exa retrieved {url}, not the demanded {challenge.metadata_url}"
        )
    content = str(entry.get("content") or "")
    if len(content) < MIN_FETCH_CONTENT_CHARS:
        raise CapabilityProbeError(
            f"web_fetch_exa returned {len(content)} characters; a real metadata document is "
            f"at least {MIN_FETCH_CONTENT_CHARS}"
        )
    declared_digest = str(entry.get("content_sha256") or "").strip().lower()
    observed_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if declared_digest != observed_digest:
        raise CapabilityProbeError(
            "web_fetch_exa content does not match its own declared sha256; the evidence is "
            "internally inconsistent"
        )
    version = _require_version(entry.get("reported_version"), "web_fetch_exa.reported_version")
    if version not in content:
        raise CapabilityProbeError(
            "web_fetch_exa reported a version that does not appear in the document it fetched"
        )
    serial = entry.get("reported_last_serial")
    if not isinstance(serial, int) or isinstance(serial, bool) or serial < MIN_RELEASE_SERIAL:
        raise CapabilityProbeError(
            "web_fetch_exa must report the document's last_serial as an integer of at least "
            f"{MIN_RELEASE_SERIAL}"
        )
    if str(serial) not in content:
        raise CapabilityProbeError(
            "web_fetch_exa reported a last_serial absent from the document it fetched"
        )
    return version


_TOOL_VALIDATORS = {
    "WebSearch": _validate_websearch,
    "web_search_exa": _validate_exa_search,
    "web_fetch_exa": _validate_exa_fetch,
}


def validate_capability_evidence(
    *,
    nonce: str,
    evidence: Mapping[str, Any],
    requested_at: datetime,
    deadline_seconds: int = 120,
    now: Optional[datetime] = None,
) -> JsonObject:
    """Turn author-session evidence into a receipt, or refuse it.

    Parameters
    ----------
    nonce:
        Doctor-minted probe nonce.
    evidence:
        Session-reported evidence: ``{"tools": {<tool>: {...}}, ...}``.
    requested_at:
        Doctor request instant, anchoring the freshness window.
    deadline_seconds:
        Freshness window width.
    now:
        Injectable observation instant.

    Returns
    -------
    dict[str, Any]
        Receipt payload the doctor accepts, carrying one nonce-bound entry per required
        tool plus the full evidence for audit.

    Raises
    ------
    CapabilityProbeError
        When any required tool is missing, its evidence is not tool-shaped, the reports
        disagree, or the work happened outside the freshness window. Every arm refuses;
        none degrades to a pass.
    """

    challenge = derive_challenge(nonce)
    raw_tools = evidence.get("tools")
    if not isinstance(raw_tools, Mapping):
        raise CapabilityProbeError("capability evidence carries no per-tool observations")
    # A tool's registered name is namespaced and the namespace depends on how the session
    # was launched, so evidence keyed `mcp__exa__web_search_exa` is the same claim as
    # evidence keyed `web_search_exa`. Resolving the spelling here means a namespace change
    # can never masquerade as a missing tool -- and nothing below is relaxed: every entry
    # still has to carry live, tool-shaped, nonce-bound, corroborated evidence.
    resolved = _canonicalize_tool_evidence(raw_tools)
    tools = {name: entry for name, (_registered, entry) in resolved.items()}
    missing = [tool for tool in REQUIRED_AUTHOR_TOOLS if tool not in tools]
    if missing:
        raise CapabilityProbeError(f"capability evidence omits required tools {missing}")
    declared_challenge = str(evidence.get("challenge_id") or "")
    if declared_challenge and declared_challenge != challenge.challenge_id:
        raise CapabilityProbeError(
            "capability evidence answers a different challenge than this nonce demands"
        )

    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    window_end = requested_at + timedelta(seconds=int(deadline_seconds))
    if not requested_at <= observed <= window_end:
        raise CapabilityProbeError("capability evidence was assembled outside the probe window")

    versions: dict[str, str] = {}
    entries: list[JsonObject] = []
    url_sets: dict[str, tuple[str, ...]] = {}
    for tool in REQUIRED_AUTHOR_TOOLS:
        entry = tools[tool]
        if not isinstance(entry, Mapping):
            raise CapabilityProbeError(f"{tool} evidence is not an object")
        if entry.get("nonce") != nonce:
            raise CapabilityProbeError(f"{tool} evidence does not echo the probe nonce")
        exercised_at = _parse_instant(entry.get("observed_at"), f"{tool}.observed_at")
        if not requested_at <= exercised_at <= window_end:
            raise CapabilityProbeError(f"{tool} was exercised outside the probe window")
        versions[tool] = _TOOL_VALIDATORS[tool](entry, challenge)
        results = entry.get("results")
        if isinstance(results, list):
            url_sets[tool] = tuple(
                str(item.get("url")) for item in results if isinstance(item, Mapping)
            )

    if url_sets.get("WebSearch") and url_sets.get("WebSearch") == url_sets.get("web_search_exa"):
        raise CapabilityProbeError(
            "WebSearch and web_search_exa returned byte-identical result lists; one tool's "
            "output was reused for the other"
        )
    distinct = set(versions.values())
    if len(distinct) != 1:
        raise CapabilityProbeError(
            f"the required tools disagree about {challenge.package}: {sorted(distinct)}"
        )

    for tool in REQUIRED_AUTHOR_TOOLS:
        registered = resolved[tool][0]
        entries.append(
            {
                "tool": tool,
                # Audit value: which namespace actually served this campaign's author path.
                # It is bound into the receipt digest rather than left as loose metadata.
                "registered_tool_name": registered,
                "nonce": nonce,
                "exercised": True,
                "receipt": stable_hash(
                    {
                        "nonce": nonce,
                        "tool": tool,
                        "registered_tool_name": registered,
                        "challenge_id": challenge.challenge_id,
                        "evidence": _canonical_tool_evidence(tools[tool]),
                    }
                ),
            }
        )
    return {
        "format": CAPABILITY_RECEIPT_FORMAT,
        "nonce": nonce,
        "challenge_id": challenge.challenge_id,
        "challenge": challenge.to_dict(),
        "answer": {"version": sorted(distinct)[0], "package": challenge.package},
        "completed_at": observed.isoformat().replace("+00:00", "Z"),
        "receipts": entries,
        "evidence": _bounded_evidence(
            {
                name: {**entry, "registered_tool_name": registered}
                for name, (registered, entry) in resolved.items()
            }
        ),
    }


def _canonical_tool_evidence(entry: Mapping[str, Any]) -> JsonObject:
    """Return the digest-bearing subset of one tool's evidence.

    Parameters
    ----------
    entry:
        Per-tool evidence object.

    Returns
    -------
    dict[str, Any]
        Fields that identify what the tool actually returned.
    """

    keys = ("url", "query", "observed_at", "reported_version", "reported_last_serial")
    canonical: JsonObject = {key: entry.get(key) for key in keys if entry.get(key) is not None}
    results = entry.get("results")
    if isinstance(results, list):
        canonical["results"] = [
            str(item.get("url")) for item in results if isinstance(item, Mapping)
        ]
    if entry.get("content_sha256") is not None:
        canonical["content_sha256"] = str(entry["content_sha256"])
    return canonical


def _bounded_evidence(tools: Mapping[str, Any]) -> JsonObject:
    """Return audit-retained evidence with oversized bodies trimmed.

    Parameters
    ----------
    tools:
        Per-tool evidence objects.

    Returns
    -------
    dict[str, Any]
        Evidence with document bodies replaced by their length and digest.
    """

    bounded: JsonObject = {}
    for tool, entry in tools.items():
        if not isinstance(entry, Mapping):
            continue
        trimmed = {key: value for key, value in entry.items() if key != "content"}
        content = entry.get("content")
        if isinstance(content, str):
            trimmed["content_chars"] = len(content)
            trimmed["content_head"] = content[:500]
        bounded[str(tool)] = trimmed
    return bounded


def required_tool_names() -> Sequence[str]:
    """Return the tools the doctor requires receipts for.

    Returns
    -------
    Sequence[str]
        Required tool names.
    """

    return REQUIRED_AUTHOR_TOOLS
