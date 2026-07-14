"""Literal evidence, locator, and claim-coverage validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

from menagerie.crawler.fetcher import cas_path
from menagerie.crawler.identity import hash_bytes


class EvidenceValidationError(ValueError):
    """Raised when literal evidence is altered, missing, or uncovered."""


@dataclass(frozen=True)
class EvidenceValidationReport:
    """Validated evidence coverage.

    Parameters
    ----------
    excerpt_count:
        Number of validated literal excerpts.
    supported_claims:
        Claim categories named by at least one valid excerpt.
    family_grounded:
        Whether at least one excerpt is explicitly family-level.
    """

    excerpt_count: int
    supported_claims: frozenset[str]
    family_grounded: bool


def validate_evidence(
    evidence: Mapping[str, Any],
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    required_claims: Iterable[str],
    *,
    cas_root: Union[str, Path, None] = None,
    require_family_grounding: bool = False,
) -> EvidenceValidationReport:
    """Validate verbatim excerpts and exhaustive claim-category coverage.

    Parameters
    ----------
    evidence:
        Model ``evidence`` block.
    source_manifest:
        A manifest wrapper containing ``sources`` or a direct source list.
    required_claims:
        Every gated claim category that must have literal support.
    cas_root:
        Optional CAS root used when manifests do not contain ``cas_path``.
    require_family_grounding:
        Whether a family-level grounding excerpt is mandatory.

    Returns
    -------
    EvidenceValidationReport
        Validated support summary.

    Raises
    ------
    EvidenceValidationError
        If an excerpt, locator, source, hash, or coverage assertion is invalid.
    """

    excerpts = evidence.get("excerpts")
    if not isinstance(excerpts, list):
        raise EvidenceValidationError("evidence.excerpts must be a list")
    sources = _source_index(source_manifest)
    supported: set[str] = set()
    evidence_ids: set[str] = set()
    family_grounded = False
    for excerpt in excerpts:
        if not isinstance(excerpt, Mapping):
            raise EvidenceValidationError("every evidence excerpt must be an object")
        evidence_id = _nonempty(excerpt.get("evidence_id"), "evidence_id")
        if evidence_id in evidence_ids:
            raise EvidenceValidationError(f"duplicate evidence_id: {evidence_id}")
        evidence_ids.add(evidence_id)
        source_id = _nonempty(excerpt.get("source_id"), f"{evidence_id}.source_id")
        source = sources.get(source_id)
        if source is None:
            raise EvidenceValidationError(f"{evidence_id} references unknown source {source_id}")
        locator = _nonempty(excerpt.get("locator"), f"{evidence_id}.locator")
        text = _nonempty(excerpt.get("text"), f"{evidence_id}.text")
        actual_text_hash = hash_bytes(text.encode("utf-8"))
        if excerpt.get("text_sha256") != actual_text_hash:
            raise EvidenceValidationError(
                f"{evidence_id} text_sha256 does not match the verbatim UTF-8 bytes"
            )
        supports = excerpt.get("supports")
        if (
            not isinstance(supports, list)
            or not supports
            or not all(isinstance(value, str) and value.strip() for value in supports)
        ):
            raise EvidenceValidationError(f"{evidence_id}.supports must be non-empty strings")
        content = _read_source(source, cas_root)
        _validate_locator(evidence_id, locator, text.encode("utf-8"), content)
        supported.update(supports)
        family_level = excerpt.get("family_level")
        if not isinstance(family_level, bool):
            raise EvidenceValidationError(f"{evidence_id}.family_level must be boolean")
        family_grounded = family_grounded or family_level

    required = {claim for claim in required_claims if claim}
    missing = required - supported
    if missing:
        raise EvidenceValidationError(f"ungrounded claim categories: {sorted(missing)}")
    coverage = evidence.get("coverage")
    if not isinstance(coverage, Mapping):
        raise EvidenceValidationError("evidence.coverage must be an object")
    declared_missing = coverage.get("missing_support")
    if declared_missing != [] or coverage.get("all_agent_fields_have_support") is not True:
        raise EvidenceValidationError("evidence coverage does not declare complete support")
    if require_family_grounding and not family_grounded:
        raise EvidenceValidationError("family-level grounding is required")
    if coverage.get("family_grounding_complete") is True and not family_grounded:
        raise EvidenceValidationError(
            "family grounding is declared complete without a family excerpt"
        )
    return EvidenceValidationReport(len(excerpts), frozenset(supported), family_grounded)


def evidence_ids(evidence: Mapping[str, Any]) -> frozenset[str]:
    """Return all unique evidence identifiers.

    Parameters
    ----------
    evidence:
        Model evidence block.

    Returns
    -------
    frozenset[str]
        Exact excerpt identifiers.

    Raises
    ------
    EvidenceValidationError
        If an identifier is missing or duplicated.
    """

    excerpts = evidence.get("excerpts", [])
    if not isinstance(excerpts, list):
        raise EvidenceValidationError("evidence.excerpts must be a list")
    result: set[str] = set()
    for excerpt in excerpts:
        if not isinstance(excerpt, Mapping):
            raise EvidenceValidationError("every evidence excerpt must be an object")
        value = _nonempty(excerpt.get("evidence_id"), "evidence_id")
        if value in result:
            raise EvidenceValidationError(f"duplicate evidence_id: {value}")
        result.add(value)
    return frozenset(result)


def _source_index(
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> dict[str, Mapping[str, Any]]:
    """Index source manifests by source identifier.

    Parameters
    ----------
    source_manifest:
        Wrapper or direct manifest sequence.

    Returns
    -------
    dict[str, Mapping[str, Any]]
        Source identifier lookup.
    """

    raw_sources: object
    if isinstance(source_manifest, Mapping):
        raw_sources = source_manifest.get("sources")
        if raw_sources is None and "source_id" in source_manifest:
            raw_sources = [source_manifest]
    else:
        raw_sources = source_manifest
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        raise EvidenceValidationError("source manifest must contain a source list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for source in raw_sources:
        if not isinstance(source, Mapping):
            raise EvidenceValidationError("every source manifest must be an object")
        source_id = _nonempty(source.get("source_id"), "source.source_id")
        if source_id in indexed:
            raise EvidenceValidationError(f"duplicate source_id: {source_id}")
        indexed[source_id] = source
    return indexed


def _read_source(source: Mapping[str, Any], cas_root: Union[str, Path, None]) -> bytes:
    """Read and re-verify exact source bytes from the CAS.

    Parameters
    ----------
    source:
        Source manifest row.
    cas_root:
        Fallback CAS root.

    Returns
    -------
    bytes
        Verified source bytes.

    Raises
    ------
    EvidenceValidationError
        If content is absent or no longer hash-bound.
    """

    digest = source.get("content_sha256")
    if not isinstance(digest, str):
        raise EvidenceValidationError("source content_sha256 is missing")
    path_value = source.get("cas_path")
    if isinstance(path_value, str) and path_value:
        path = Path(path_value)
    elif cas_root is not None:
        path = cas_path(cas_root, digest)
    else:
        raise EvidenceValidationError("source manifest has no CAS path")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise EvidenceValidationError(f"cannot read source CAS object {path}: {exc}") from exc
    if hash_bytes(content) != digest:
        raise EvidenceValidationError(f"source CAS object does not match {digest}")
    return content


def _validate_locator(evidence_id: str, locator: str, text: bytes, content: bytes) -> None:
    """Validate an excerpt against an exact byte locator or source membership.

    Parameters
    ----------
    evidence_id:
        Excerpt identifier used in errors.
    locator:
        Locator string. ``bytes:start-end`` receives strict byte-range checking.
    text:
        Claimed verbatim excerpt bytes.
    content:
        Exact fetched source bytes.

    Raises
    ------
    EvidenceValidationError
        If the excerpt is absent or differs at the byte locator.
    """

    if locator.startswith("bytes:"):
        try:
            start_text, end_text = locator.removeprefix("bytes:").split("-", 1)
            start, end = int(start_text), int(end_text)
        except (ValueError, TypeError) as exc:
            raise EvidenceValidationError(f"{evidence_id} has an invalid byte locator") from exc
        if start < 0 or end < start or content[start:end] != text:
            raise EvidenceValidationError(f"{evidence_id} does not exist verbatim at {locator}")
    elif text not in content:
        raise EvidenceValidationError(
            f"{evidence_id} excerpt is not verbatim in its fetched source"
        )


def _nonempty(value: object, field: str) -> str:
    """Return a stripped non-empty string.

    Parameters
    ----------
    value:
        Candidate field value.
    field:
        Field name used in errors.

    Returns
    -------
    str
        Original non-empty string.

    Raises
    ------
    EvidenceValidationError
        If the value is not a non-empty string.
    """

    if not isinstance(value, str) or not value.strip():
        raise EvidenceValidationError(f"{field} must be a non-empty string")
    return value
