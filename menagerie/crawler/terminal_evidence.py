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

Excerpts ARRIVE through the declared ``evidence_records`` channel on the terminal
payload -- a validated part of the author-result contract, so an author that says
nothing is a schema-visible silence rather than a missing file nobody declared.
The channel is where the claim arrives; it is never why the claim is believed.
An author that quotes text absent from the frozen bytes is refused on the
declared channel exactly as it would be anywhere else, and a declared channel
that fails to ground does NOT fall back to a file -- otherwise the file would
launder what the contract just rejected.

Resolution is PER RECORD, not all-or-nothing. Every excerpt shown was matched
verbatim against frozen bytes; every declared ID that was not lands in
``unresolved_evidence_ids`` and is shown to nobody. The two sets partition the
declared IDs exactly, so a pack can never present an unverified record as
verified, and the resolution stops being ``grounded`` the instant one ID fails.
Two floors bound the partial verdict: nothing verified is a hard ``unresolved``,
and so is a verified subset in which no row even claims the terminal predicate,
because grounded decoration around a dead predicate is not partial grounding.
See :func:`_settle` for why the previous all-or-nothing rule paid authors to
cite LESS evidence.

``CANDIDATE_EVIDENCE_FILENAMES`` remains as a compatibility fallback for attempt
directories written before the channel existed, and is consulted ONLY when the
payload declares no records at all. It is not the contract; it is a reader for
history.
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
PARTIALLY_GROUNDED = "partially-grounded"
UNRESOLVED = "unresolved"

#: Every resolution a terminal evidence pack may declare.
TERMINAL_EVIDENCE_RESOLUTIONS = frozenset({GROUNDED, PARTIALLY_GROUNDED, UNRESOLVED})

#: Closed vocabulary naming where the excerpts a resolution judged arrived from.
CHANNEL_DECLARED = "declared-payload"
CHANNEL_ATTEMPT_DIRECTORY = "attempt-directory"
CHANNEL_MACHINE_DISCOVERY = "machine-discovery"
CHANNEL_NONE = "none"

#: Fields an excerpt record must carry before the machine can re-derive it.
REQUIRED_EXCERPT_FIELDS = ("evidence_id", "source_id", "locator", "text")

#: Per-record keys only the machine may own. An author cannot compute a digest,
#: so it is never asked for one; declaring one anyway is refused rather than
#: quietly trusted.
MACHINE_OWNED_EXCERPT_FIELDS = ("text_sha256", "evidence_identity", "content_sha256")


@dataclass(frozen=True)
class TerminalEvidenceResolution:
    """Outcome of re-deriving an author's cited excerpts from frozen bytes.

    Parameters
    ----------
    resolution:
        ``grounded`` when every declared evidence ID resolved to a literal
        excerpt the machine matched verbatim against its frozen source;
        ``partially-grounded`` when a strict subset did AND that subset still
        includes an excerpt claiming the terminal predicate, with every gap
        named; ``unresolved`` when nothing usable verified -- none did, or the
        predicate-bearing rows themselves all failed.
    excerpts:
        Exact verified excerpt records, in declared order. Every row here was
        matched byte-for-byte against frozen bytes -- an unverified row is never
        presented as evidence at any resolution. Empty when unresolved.
    unresolved_evidence_ids:
        Declared evidence IDs with no inspectable, re-derived record. Together
        with the IDs on ``excerpts`` this partitions the declared set exactly:
        every declared ID lands in one bucket and never in both.
    reason:
        Machine-written explanation naming each gap, for any outcome that is not
        fully grounded.
    channel:
        Closed name of the channel whose records this verdict judged, so a
        reader can tell a contract-declared grounding from a historical
        attempt-directory one and from having nothing to judge at all.
    """

    resolution: str
    excerpts: tuple[JsonObject, ...]
    unresolved_evidence_ids: tuple[str, ...]
    reason: Optional[str]
    channel: str = CHANNEL_NONE

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
    declared_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> TerminalEvidenceResolution:
    """Re-derive every cited terminal excerpt from frozen source bytes.

    The declared channel is authoritative: when the terminal payload carries
    ``evidence_records``, those records are what gets judged, and their verdict
    stands. A declared channel that fails to ground never falls through to the
    attempt-directory reader, because that would let a file re-supply exactly
    what dereference just refused.

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
    declared_records:
        Contract-declared excerpt records from the terminal payload.

    Returns
    -------
    TerminalEvidenceResolution
        Verified excerpts, or an explicit unresolved declaration.
    """

    declared = tuple(str(value) for value in evidence_ids)
    if not declared:
        return TerminalEvidenceResolution(
            UNRESOLVED, (), (), "the terminal recommendation cites no evidence at all", CHANNEL_NONE
        )
    discovery = _resolve_machine_discovery(source_manifest, declared, predicate)
    if discovery is not None:
        return discovery
    cas_root = author_root / SOURCE_CAS_DIRNAME
    if declared_records:
        verified, reasons = _verify_records(
            list(declared_records), declared, source_manifest, cas_root, CHANNEL_DECLARED
        )
        return _settle(verified, reasons, declared, CHANNEL_DECLARED, predicate)
    return _resolve_author_excerpts(source_manifest, declared, author_root, predicate)


def _settle(
    verified: Sequence[JsonObject],
    reasons: Sequence[str],
    declared: tuple[str, ...],
    channel: str,
    predicate: str,
) -> TerminalEvidenceResolution:
    """Turn one channel's per-record outcome into a resolution.

    Resolution used to be all-or-nothing: one ungrounded ID discarded every
    excerpt that DID re-derive. Pilot model ``m9617`` is the proof that this is
    not a tripwire but an inversion of the incentive it was meant to create. It
    emitted an honest ``BLOCKED`` citing thirteen records; ten grounded
    byte-for-byte, including the one carrying the ``blocked-prerequisite``
    predicate, and three cited a response the broker never froze. All ten were
    thrown away, the checker was handed nothing to read, and the model died
    ``terminal-disposition-unverifiable``. The author would have SURVIVED by
    citing fewer IDs -- a system that wants more grounding paying authors to
    ground less.

    Per-record settlement removes that inversion without conceding anything to
    the padding argument it was defending against. A padded ID cannot launder:
    it never becomes an excerpt, it is named in ``unresolved_evidence_ids``, and
    the resolution stops being ``grounded`` the moment one ID fails. What the
    partial verdict buys is that the nine good citations beside a bad one are
    still shown, under a name that says plainly they are not the whole pack.

    Two floors keep the partial verdict honest. Nothing verified is a hard
    ``unresolved``. And a partial pack must still carry the terminal predicate
    itself: if no verified row's ``supports`` even claims the predicate the
    recommendation rests on -- because the predicate-bearing rows are exactly the
    ones that failed -- then the pack has grounded only decoration, and grounded
    decoration around an ungrounded predicate is not partial grounding. It stays
    ``unresolved`` with an empty excerpt tuple, so padding-in-reverse (dressing a
    dead predicate in verified trimmings) buys nothing.

    Parameters
    ----------
    verified:
        Rows this channel re-derived from frozen bytes, in declared order.
    reasons:
        One named reason per declared ID that did not re-derive.
    declared:
        Exact declared evidence identities, in declared order.
    channel:
        Closed channel name whose records were judged.
    predicate:
        Closed typed terminal predicate the evidence must still carry for a
        partial verdict to stand.

    Returns
    -------
    TerminalEvidenceResolution
        Grounded, partially grounded, or unresolved outcome.
    """

    if len(verified) == len(declared):
        return TerminalEvidenceResolution(GROUNDED, tuple(verified), (), None, channel)
    grounded_ids = {str(row["evidence_id"]) for row in verified}
    unresolved = tuple(value for value in declared if value not in grounded_ids)
    detail = "; ".join(reasons)
    if not verified:
        return TerminalEvidenceResolution(
            UNRESOLVED,
            (),
            unresolved,
            f"no declared evidence_record re-derived from its frozen source: {detail}",
            channel,
        )
    if not _any_supports_predicate(verified, predicate):
        return TerminalEvidenceResolution(
            UNRESOLVED,
            (),
            unresolved,
            f"no verified excerpt claims the terminal predicate {predicate!r}: {detail}",
            channel,
        )
    return TerminalEvidenceResolution(
        PARTIALLY_GROUNDED,
        tuple(verified),
        unresolved,
        f"{len(verified)} of {len(declared)} declared evidence IDs re-derived from frozen "
        f"bytes; the rest are named here and are shown to no one: {detail}",
        channel,
    )


def _any_supports_predicate(verified: Sequence[JsonObject], predicate: str) -> bool:
    """Return whether any verified excerpt claims to support the terminal predicate.

    The ``supports`` list is the author's claim about what each excerpt shows;
    whether the text actually supports it stays the checker's judgment. This
    floor only prevents the degenerate partial pack whose grounded rows never
    even claim the predicate the terminal recommendation rests on.

    Parameters
    ----------
    verified:
        Excerpt rows already re-derived from frozen source bytes.
    predicate:
        Closed typed terminal predicate.

    Returns
    -------
    bool
        Whether at least one verified row's ``supports`` names the predicate.
    """

    return any(
        isinstance(row.get("supports"), list) and predicate in row["supports"]
        for row in verified
    )


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
                "origin": CHANNEL_MACHINE_DISCOVERY,
            },
        ),
        (),
        None,
        CHANNEL_MACHINE_DISCOVERY,
    )


def _resolve_author_excerpts(
    source_manifest: Mapping[str, Any],
    declared: tuple[str, ...],
    author_root: Path,
    predicate: str,
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
    predicate:
        Closed typed terminal predicate handed through to settlement.

    Returns
    -------
    TerminalEvidenceResolution
        Verified excerpts, or an explicit unresolved declaration.
    """

    no_channel = "the terminal payload declared no evidence_records"
    candidates = _candidate_paths(author_root, TERMINAL_EVIDENCE_FILENAME)
    if not candidates:
        return TerminalEvidenceResolution(
            UNRESOLVED,
            (),
            declared,
            f"{no_channel} and no frozen {TERMINAL_EVIDENCE_FILENAME} was published "
            f"under {author_root}",
            CHANNEL_NONE,
        )
    cas_root = author_root / SOURCE_CAS_DIRNAME
    last_reason = ""
    best: Optional[TerminalEvidenceResolution] = None
    for path in candidates:
        record = _read_json_object(path)
        if record is None:
            last_reason = f"{path} is not one readable JSON object"
            continue
        excerpts = record.get("excerpts")
        if not isinstance(excerpts, list):
            last_reason = f"{path} carries no excerpt list"
            continue
        verified, reasons = _verify_records(
            excerpts, declared, source_manifest, cas_root, CHANNEL_ATTEMPT_DIRECTORY
        )
        settled = _settle(verified, reasons, declared, CHANNEL_ATTEMPT_DIRECTORY, predicate)
        if settled.grounded:
            return settled
        last_reason = f"{path}: " + "; ".join(reasons)
        # A fully grounded later candidate still wins, so keep scanning; a
        # partial one is only used when no candidate grounds outright.
        if best is None and settled.excerpts:
            best = TerminalEvidenceResolution(
                settled.resolution,
                settled.excerpts,
                settled.unresolved_evidence_ids,
                f"{path}: {settled.reason}",
                CHANNEL_ATTEMPT_DIRECTORY,
            )
    if best is not None:
        return best
    return TerminalEvidenceResolution(
        UNRESOLVED, (), declared, f"{no_channel}; {last_reason}", CHANNEL_ATTEMPT_DIRECTORY
    )


def _verify_records(
    records: Sequence[Any],
    declared: tuple[str, ...],
    source_manifest: Mapping[str, Any],
    cas_root: Path,
    origin: str,
) -> tuple[list[JsonObject], list[str]]:
    """Match one channel's excerpt records verbatim against their frozen sources.

    This is the single dereference point every channel goes through. A record is
    only ever returned after its claimed text was found byte-for-byte in the
    hash-bound source bytes; the digest on the returned row is recomputed here
    from those bytes, so an author-supplied one can neither be required nor
    believed.

    Parameters
    ----------
    records:
        Candidate excerpt records from one channel.
    declared:
        Exact declared evidence identities, in declared order.
    source_manifest:
        Frozen source manifest bound by the terminal recommendation.
    cas_root:
        Content-addressed store holding the fetched source bytes.
    origin:
        Closed channel name stamped onto every verified row.

    Returns
    -------
    tuple[list[dict[str, Any]], list[str]]
        Verified rows in declared order, and one named reason per gap.
    """

    by_id = {
        str(record["evidence_id"]): dict(record)
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("evidence_id"), str)
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
        row = {key: value for key, value in excerpt.items() if key not in MACHINE_OWNED_EXCERPT_FIELDS}
        row["text_sha256"] = hash_bytes(str(excerpt["text"]).encode("utf-8"))
        row["origin"] = origin
        verified.append(row)
    return verified, reasons


def resolve_terminal_license_record(
    *,
    source_manifest: Mapping[str, Any],
    license_record: Optional[Mapping[str, Any]],
    author_root: Path,
) -> Optional[JsonObject]:
    """Ground one declared license excerpt against its frozen source bytes.

    The returned row is presentation only. ``license_identity`` stays derived by
    :func:`menagerie.crawler.author_dispatch.derive_terminal_license_disposition`
    from facts the machine owns; quoting license text gives the author's reading a
    declared, inspectable home without moving that identity to the author.

    Parameters
    ----------
    source_manifest:
        Frozen source manifest bound by the terminal recommendation.
    license_record:
        Declared license excerpt from the terminal payload, when present.
    author_root:
        Private staging root for one model's author round trips.

    Returns
    -------
    dict[str, Any] | None
        Grounded or explicitly ungrounded license row, or ``None`` when the
        payload declares no license text at all.
    """

    if not isinstance(license_record, Mapping) or not license_record:
        return None
    row = dict(license_record)
    if not all(
        isinstance(row.get(field), str) and row.get(field)
        for field in ("source_id", "locator", "text")
    ):
        return {
            "resolution": UNRESOLVED,
            "reason": "the declared license record is not an inspectable excerpt",
        }
    probe = {**row, "evidence_id": "license-record"}
    try:
        _verify_against_frozen_source(probe, source_manifest, author_root / SOURCE_CAS_DIRNAME)
    except EvidenceValidationError as exc:
        return {
            "resolution": UNRESOLVED,
            "reason": f"the declared license record did not re-derive from its frozen source: {exc}",
        }
    grounded = {key: value for key, value in row.items() if key not in MACHINE_OWNED_EXCERPT_FIELDS}
    grounded["text_sha256"] = hash_bytes(str(row["text"]).encode("utf-8"))
    grounded["resolution"] = GROUNDED
    grounded["origin"] = CHANNEL_DECLARED
    return grounded


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
