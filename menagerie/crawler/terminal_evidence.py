"""Frozen terminal evidence and license record resolution for checker envelopes.

A terminal recommendation -- ``DEFER_RECOMMENDATION``, ``SKIP_RECOMMENDATION``,
or ``BLOCKED`` -- declares ``evidence_ids``, ``evidence_identity``, and
``license_identity``. Those are *identifiers*. An independent checker cannot
verify a claim from an identifier: it needs the literal excerpt, the source
locator, and enough canonical bytes to recompute the identity it is asked to
trust. Handing it identifiers alone guarantees a ``cannot-verify`` verdict, and
synthesizing plausible-looking excerpt rows to fill the gap is fabrication --
the exact failure mode the whole evidence design exists to prevent.

This module resolves the author's own frozen records and binds them to the
declared identities. A record that does not recompute to its declared identity
is NOT evidence and is never presented as such: the resolution is reported as
``unresolved``, naming the observed and declared digests, so the gap is visible
to the checker instead of being papered over.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.identity import compute_evidence_identity, stable_hash
from menagerie.crawler.models import JsonObject

TERMINAL_EVIDENCE_FILENAME = "evidence-pack.json"
TERMINAL_LICENSE_FILENAME = "license-pack.json"
ATTEMPTS_DIRNAME = "attempts"

GROUNDED = "grounded"
UNRESOLVED = "unresolved"

#: Fields an excerpt record must carry before a checker can inspect the claim it
#: grounds. ``supports`` is included because the typed terminal predicate is
#: exactly what the checker must see the excerpt support.
REQUIRED_EXCERPT_FIELDS = ("evidence_id", "source_id", "locator", "text", "text_sha256", "supports")


@dataclass(frozen=True)
class TerminalEvidenceResolution:
    """Outcome of binding an author's frozen evidence records to its identity.

    Parameters
    ----------
    resolution:
        ``grounded`` when every declared evidence ID resolved to a literal,
        identity-bound excerpt record; ``unresolved`` otherwise.
    excerpts:
        Exact frozen excerpt records, in declared order. Empty when unresolved.
    unresolved_evidence_ids:
        Declared evidence IDs with no inspectable record.
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


@dataclass(frozen=True)
class TerminalLicenseResolution:
    """Outcome of binding an author's frozen license record to its identity.

    Parameters
    ----------
    resolution:
        ``grounded`` or ``unresolved``.
    record:
        Exact frozen license record, or ``None`` when unresolved.
    reason:
        Machine-written explanation of an unresolved outcome.
    """

    resolution: str
    record: Optional[JsonObject]
    reason: Optional[str]

    @property
    def grounded(self) -> bool:
        """Return whether the license record recomputes to its declared identity."""

        return self.resolution == GROUNDED


def resolve_terminal_evidence(
    *,
    author_root: Path,
    evidence_ids: Sequence[str],
    evidence_identity: str,
) -> TerminalEvidenceResolution:
    """Bind the author's frozen excerpt records to its declared evidence identity.

    Parameters
    ----------
    author_root:
        Private staging root for one model's author round trips.
    evidence_ids:
        Exact evidence identities the recommendation declares.
    evidence_identity:
        Exact evidence-pack identity the recommendation declares.

    Returns
    -------
    TerminalEvidenceResolution
        Grounded records, or an explicit unresolved declaration.
    """

    declared = tuple(str(value) for value in evidence_ids)
    candidates = _candidate_paths(author_root, TERMINAL_EVIDENCE_FILENAME)
    if not candidates:
        return TerminalEvidenceResolution(
            UNRESOLVED,
            (),
            declared,
            f"no frozen {TERMINAL_EVIDENCE_FILENAME} was published under {author_root}",
        )
    last_reason = ""
    for path in candidates:
        record = _read_json_object(path)
        if record is None:
            last_reason = f"{path} is not one readable JSON object"
            continue
        excerpts = record.get("excerpts")
        if not isinstance(excerpts, list) or not all(
            isinstance(excerpt, Mapping) for excerpt in excerpts
        ):
            last_reason = f"{path} carries no excerpt list"
            continue
        observed = compute_evidence_identity(excerpts)
        if observed != evidence_identity:
            # An unbound record is not evidence. Presenting it anyway would let
            # any excerpt list stand in for the one the author actually hashed.
            last_reason = (
                f"{path} recomputes to {observed}, not the declared "
                f"evidence_identity {evidence_identity}"
            )
            continue
        by_id = {
            str(excerpt["evidence_id"]): dict(excerpt)
            for excerpt in excerpts
            if isinstance(excerpt, Mapping) and isinstance(excerpt.get("evidence_id"), str)
        }
        missing = tuple(
            evidence_id
            for evidence_id in declared
            if evidence_id not in by_id or not _is_inspectable(by_id[evidence_id])
        )
        if missing:
            return TerminalEvidenceResolution(
                UNRESOLVED,
                (),
                missing,
                f"{path} binds its identity but supplies no inspectable excerpt for "
                f"{', '.join(missing)}",
            )
        return TerminalEvidenceResolution(
            GROUNDED, tuple(by_id[evidence_id] for evidence_id in declared), (), None
        )
    return TerminalEvidenceResolution(UNRESOLVED, (), declared, last_reason)


def resolve_terminal_license(
    *, author_root: Path, license_identity: str
) -> TerminalLicenseResolution:
    """Bind the author's frozen license record to its declared license identity.

    Parameters
    ----------
    author_root:
        Private staging root for one model's author round trips.
    license_identity:
        Exact license-disposition identity the recommendation declares.

    Returns
    -------
    TerminalLicenseResolution
        Grounded record, or an explicit unresolved declaration.
    """

    candidates = _candidate_paths(author_root, TERMINAL_LICENSE_FILENAME)
    if not candidates:
        return TerminalLicenseResolution(
            UNRESOLVED,
            None,
            f"no frozen {TERMINAL_LICENSE_FILENAME} was published under {author_root}",
        )
    last_reason = ""
    for path in candidates:
        record = _read_json_object(path)
        if record is None:
            last_reason = f"{path} is not one readable JSON object"
            continue
        observed = stable_hash(record)
        if observed != license_identity:
            last_reason = (
                f"{path} recomputes to {observed}, not the declared "
                f"license_identity {license_identity}"
            )
            continue
        return TerminalLicenseResolution(GROUNDED, record, None)
    return TerminalLicenseResolution(UNRESOLVED, None, last_reason)


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


def _is_inspectable(excerpt: Mapping[str, Any]) -> bool:
    """Return whether one excerpt record carries everything a checker must read.

    Parameters
    ----------
    excerpt:
        Candidate frozen excerpt record.

    Returns
    -------
    bool
        Whether the literal text, its locator, its digest, and its supported
        claims are all present.
    """

    for field in REQUIRED_EXCERPT_FIELDS:
        value = excerpt.get(field)
        if field == "supports":
            if not isinstance(value, list) or not value:
                return False
            if not all(isinstance(claim, str) and claim for claim in value):
                return False
            continue
        if not isinstance(value, str) or not value:
            return False
    return True
