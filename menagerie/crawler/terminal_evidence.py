"""Literal terminal evidence resolution for the checker envelope.

A terminal recommendation -- ``DEFER_RECOMMENDATION``, ``SKIP_RECOMMENDATION``,
or ``BLOCKED`` -- is judged by an independent checker. That checker cannot verify
a claim from an identifier: it needs the literal excerpt, the source locator, and
bytes it can re-derive. Handing it identifiers alone guarantees a
``cannot-verify`` verdict, and synthesizing plausible excerpt rows to fill the
gap is fabrication -- the exact failure the evidence design exists to prevent.

The machine-derived citation table from
:func:`menagerie.crawler.author_dispatch.derive_terminal_evidence_pack` owns the
evidence IDENTITY, and correctly so: the author has no hashing primitive and must
never be asked for a digest. That table is not touched here. What this module
adds is the only thing that turns an identifier into something inspectable, under
the rule the discovery convergence established -- *a locator the machine can
verify by dereferencing may be model-supplied.* Every excerpt returned here was
read back out of content-addressed storage and matched verbatim against the
frozen source bytes. Nothing that fails that check is presented as evidence; it
is reported as an explicit, named gap instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.evidence import (
    EvidenceValidationError,
    _read_source,
    _validate_locator,
)
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.models import JsonObject

TERMINAL_EVIDENCE_FILENAME = "evidence-pack.json"
ATTEMPTS_DIRNAME = "attempts"
SOURCE_CAS_DIRNAME = "source-cas"
DISCOVERY_EVIDENCE_KIND = "discovery-evidence-v1"

GROUNDED = "grounded"
UNRESOLVED = "unresolved"

#: Fields an excerpt record must carry before the machine can re-derive it.
REQUIRED_EXCERPT_FIELDS = ("evidence_id", "source_id", "locator", "text")


@dataclass(frozen=True)
class TerminalEvidenceResolution:
    """Outcome of re-deriving an author's cited excerpts from frozen bytes.

    Parameters
    ----------
    resolution:
        ``grounded`` when every declared evidence ID resolved to a literal
        excerpt the machine matched verbatim against its frozen source;
        ``unresolved`` otherwise.
    excerpts:
        Exact verified excerpt records, in declared order. Empty when
        unresolved -- an unverified row is never presented as evidence.
    unresolved_evidence_ids:
        Declared evidence IDs with no inspectable, re-derived record.
    reason:
        Machine-written explanation of an unresolved outcome.
    """

    resolution: str
    excerpts: tuple[JsonObject, ...]
    unresolved_evidence_ids: tuple[str, ...]
    reason: Optional[str]

    @property
    def grounded(self) -> bool:
        """Return whether every declared evidence ID is literally inspectable."""

        return self.resolution == GROUNDED


def resolve_terminal_evidence(
    *,
    source_manifest: Mapping[str, Any],
    evidence_ids: Sequence[str],
    predicate: str,
    author_root: Path,
) -> TerminalEvidenceResolution:
    """Re-derive every cited terminal excerpt from frozen source bytes.

    Parameters
    ----------
    source_manifest:
        Frozen source manifest bound by the terminal recommendation.
    evidence_ids:
        Exact evidence identities the recommendation declares.
    predicate:
        Closed typed terminal predicate the evidence resolves.
    author_root:
        Private staging root for one model's author round trips.

    Returns
    -------
    TerminalEvidenceResolution
        Verified excerpts, or an explicit unresolved declaration.
    """

    declared = tuple(str(value) for value in evidence_ids)
    if not declared:
        return TerminalEvidenceResolution(
            UNRESOLVED, (), (), "the terminal recommendation cites no evidence at all"
        )
    discovery = _resolve_machine_discovery(source_manifest, declared, predicate)
    if discovery is not None:
        return discovery
    return _resolve_author_excerpts(source_manifest, declared, author_root)


def _resolve_machine_discovery(
    source_manifest: Mapping[str, Any], declared: tuple[str, ...], predicate: str
) -> Optional[TerminalEvidenceResolution]:
    """Ground a driver-derived discovery terminal from the bytes it froze.

    A machine discovery arm has no author excerpt pack: the driver itself froze
    the discovery result into content-addressed storage. Those bytes ARE the
    literal evidence, and their digest is recomputed here before they are shown,
    so no link in this chain rests on a model claim.

    Parameters
    ----------
    source_manifest:
        Frozen one-row discovery-evidence manifest.
    declared:
        Exact declared evidence identities.
    predicate:
        Closed typed terminal predicate.

    Returns
    -------
    TerminalEvidenceResolution | None
        Resolution, or ``None`` when this is not a machine discovery manifest.
    """

    sources = source_manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 1 or len(declared) != 1:
        return None
    row = sources[0]
    if not isinstance(row, Mapping) or row.get("source_kind") != DISCOVERY_EVIDENCE_KIND:
        return None
    cas_path = row.get("cas_path")
    digest = row.get("content_sha256")
    if not isinstance(cas_path, str) or not isinstance(digest, str):
        return None
    try:
        content = Path(cas_path).read_bytes()
        text = content.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return TerminalEvidenceResolution(
            UNRESOLVED,
            (),
            declared,
            f"frozen discovery evidence at {cas_path} is unreadable: {exc}",
        )
    observed = hash_bytes(content)
    if observed != digest:
        return TerminalEvidenceResolution(
            UNRESOLVED,
            (),
            declared,
            f"frozen discovery evidence at {cas_path} hashes to {observed}, not {digest}",
        )
    return TerminalEvidenceResolution(
        GROUNDED,
        (
            {
                "evidence_id": declared[0],
                "source_id": str(row.get("source_id")),
                "locator": str(row.get("url", cas_path)),
                "text": text,
                "text_sha256": observed,
                "supports": [predicate],
                "origin": "machine-discovery",
            },
        ),
        (),
        None,
    )


def _resolve_author_excerpts(
    source_manifest: Mapping[str, Any], declared: tuple[str, ...], author_root: Path
) -> TerminalEvidenceResolution:
    """Match every author-cited excerpt verbatim against its frozen source.

    Parameters
    ----------
    source_manifest:
        Frozen source manifest bound by the terminal recommendation.
    declared:
        Exact declared evidence identities.
    author_root:
        Private staging root for one model's author round trips.

    Returns
    -------
    TerminalEvidenceResolution
        Verified excerpts, or an explicit unresolved declaration.
    """

    candidates = _candidate_paths(author_root, TERMINAL_EVIDENCE_FILENAME)
    if not candidates:
        return TerminalEvidenceResolution(
            UNRESOLVED,
            (),
            declared,
            f"no frozen {TERMINAL_EVIDENCE_FILENAME} was published under {author_root}",
        )
    cas_root = author_root / SOURCE_CAS_DIRNAME
    last_reason = ""
    for path in candidates:
        record = _read_json_object(path)
        if record is None:
            last_reason = f"{path} is not one readable JSON object"
            continue
        excerpts = record.get("excerpts")
        if not isinstance(excerpts, list):
            last_reason = f"{path} carries no excerpt list"
            continue
        by_id = {
            str(excerpt["evidence_id"]): dict(excerpt)
            for excerpt in excerpts
            if isinstance(excerpt, Mapping) and isinstance(excerpt.get("evidence_id"), str)
        }
        verified: list[JsonObject] = []
        reasons: list[str] = []
        for evidence_id in declared:
            excerpt = by_id.get(evidence_id)
            if excerpt is None or not _has_required_fields(excerpt):
                reasons.append(f"{evidence_id} has no inspectable excerpt record")
                continue
            try:
                _verify_against_frozen_source(excerpt, source_manifest, cas_root)
            except EvidenceValidationError as exc:
                reasons.append(f"{evidence_id} did not re-derive from its frozen source: {exc}")
                continue
            verified.append(excerpt)
        if len(verified) != len(declared):
            last_reason = f"{path}: " + "; ".join(reasons)
            continue
        return TerminalEvidenceResolution(GROUNDED, tuple(verified), (), None)
    return TerminalEvidenceResolution(UNRESOLVED, (), declared, last_reason)


def _verify_against_frozen_source(
    excerpt: Mapping[str, Any], source_manifest: Mapping[str, Any], cas_root: Path
) -> None:
    """Require one excerpt to appear verbatim in its frozen source bytes.

    Parameters
    ----------
    excerpt:
        Candidate author excerpt record.
    source_manifest:
        Frozen source manifest bound by the terminal recommendation.
    cas_root:
        Content-addressed store holding the fetched source bytes.

    Raises
    ------
    EvidenceValidationError
        If the source is outside the frozen manifest, its bytes are absent or no
        longer hash-bound, or the claimed text is not present at its locator.
    """

    source_id = str(excerpt["source_id"])
    sources = source_manifest.get("sources")
    if not isinstance(sources, list):
        raise EvidenceValidationError("terminal source manifest has no sources")
    row = next(
        (
            source
            for source in sources
            if isinstance(source, Mapping) and source.get("source_id") == source_id
        ),
        None,
    )
    if row is None:
        raise EvidenceValidationError(f"{source_id} is outside the frozen source manifest")
    content = _read_source(row, cas_root)
    _validate_locator(
        str(excerpt["evidence_id"]),
        str(excerpt["locator"]),
        str(excerpt["text"]).encode("utf-8"),
        content,
    )


def _candidate_paths(author_root: Path, filename: str) -> tuple[Path, ...]:
    """Return every frozen record candidate, newest author attempt first.

    Parameters
    ----------
    author_root:
        Private staging root for one model's author round trips.
    filename:
        Exact frozen record filename.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Existing candidate paths.
    """

    found: list[Path] = []
    direct = author_root / filename
    if direct.is_file():
        found.append(direct)
    attempts = author_root / ATTEMPTS_DIRNAME
    if attempts.is_dir():
        for attempt in sorted(attempts.iterdir(), reverse=True):
            candidate = attempt / filename
            if candidate.is_file():
                found.append(candidate)
    return tuple(found)


def _read_json_object(path: Path) -> Optional[JsonObject]:
    """Return one frozen record object, or ``None`` when it is unreadable.

    Parameters
    ----------
    path:
        Candidate record path.

    Returns
    -------
    dict[str, Any] | None
        Parsed object, or ``None``.
    """

    try:
        if path.is_symlink():
            return None
        parsed: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(parsed) if isinstance(parsed, dict) else None


def _has_required_fields(excerpt: Mapping[str, Any]) -> bool:
    """Return whether one excerpt record carries what re-derivation needs.

    Parameters
    ----------
    excerpt:
        Candidate frozen excerpt record.

    Returns
    -------
    bool
        Whether the literal text, its locator, and its source are all present.
    """

    return all(
        isinstance(excerpt.get(field), str) and excerpt.get(field)
        for field in REQUIRED_EXCERPT_FIELDS
    )
