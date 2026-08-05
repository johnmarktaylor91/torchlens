"""Literal evidence, locator, and claim-coverage validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Any, Iterable, Mapping, Optional, Sequence, Union

from menagerie.crawler.fetcher import cas_path
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.source_broker import OUTCOME_UNFETCHABLE_BY_POLICY


POLICY_CHECKED_LINK_DISPOSITIONS = frozenset({"unfetchable-by-policy"})
POLICY_CHECKED_LINK_OUTCOMES = frozenset(
    {OUTCOME_UNFETCHABLE_BY_POLICY, "redirect-refused"}
)


class EvidenceValidationError(ValueError):
    """Raised when literal evidence is altered, missing, or uncovered."""


SMP_ZOO_PREFIX = "segmentation_models_pytorch-"
SMP_LEGACY_NAME_PREFIX = "smp_"


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
    absence_covered_claims:
        Claim categories discharged by a typed absence state rather than by an excerpt.
        Kept separate from ``supported_claims`` so a declared absence can never be
        mistaken for positive literal support when reading the report.
    """

    excerpt_count: int
    supported_claims: frozenset[str]
    family_grounded: bool
    absence_covered_claims: frozenset[str] = frozenset()


def trusted_intake_identity_mismatches(
    proposed_identity: Mapping[str, Any],
    trusted_identity: Mapping[str, Any],
    *,
    intake_name: str,
    intake_zoo: str,
) -> dict[str, dict[str, Any]]:
    """Return trusted-intake identity leaves contradicted by proposed facts.

    The trusted-intake tripwire stays exact for every field except one legacy SMP
    spelling: roster names use ``smp_<model>_<encoder>`` as their full natural-key
    token, while source-grounded R1 proposals use the encoder token as
    ``identity.variant``. That equivalence is accepted only when the intake row is
    from segmentation_models_pytorch and the trusted full token decomposes exactly
    around the proposed encoder token.

    Parameters
    ----------
    proposed_identity:
        Author-proposed ``identity`` object.
    trusted_identity:
        Trusted-intake projection for the same item.
    intake_name:
        Raw trusted intake ``name`` token.
    intake_zoo:
        Raw trusted intake ``zoo`` token.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mismatched trusted fields, keyed by field name.
    """

    return {
        field: {"proposed": proposed_identity.get(field), "trusted": trusted_value}
        for field, trusted_value in trusted_identity.items()
        if not _trusted_identity_field_matches(
            field,
            proposed_identity.get(field),
            trusted_value,
            intake_name=intake_name,
            intake_zoo=intake_zoo,
        )
    }


def _trusted_identity_field_matches(
    field: str,
    proposed_value: Any,
    trusted_value: Any,
    *,
    intake_name: str,
    intake_zoo: str,
) -> bool:
    """Return whether one trusted identity field is satisfied."""

    if proposed_value == trusted_value:
        return True
    if field != "variant" or not isinstance(proposed_value, str) or not isinstance(
        trusted_value, str
    ):
        return False
    return _smp_legacy_variant_matches(
        proposed_variant=proposed_value,
        trusted_variant=trusted_value,
        intake_name=intake_name,
        intake_zoo=intake_zoo,
    )


def _smp_legacy_variant_matches(
    *,
    proposed_variant: str,
    trusted_variant: str,
    intake_name: str,
    intake_zoo: str,
) -> bool:
    """Return whether an SMP full roster token proves the proposed encoder variant."""

    if not intake_zoo.startswith(SMP_ZOO_PREFIX):
        return False
    if trusted_variant != intake_name or not trusted_variant.startswith(SMP_LEGACY_NAME_PREFIX):
        return False
    if proposed_variant != proposed_variant.strip() or not proposed_variant:
        return False
    encoder = _smp_encoder_token(trusted_variant, proposed_variant)
    return encoder == proposed_variant


def _smp_encoder_token(trusted_variant: str, proposed_variant: str) -> Optional[str]:
    """Extract the SMP encoder suffix from a trusted roster token, if it is exact."""

    prefix = SMP_LEGACY_NAME_PREFIX
    suffix = f"_{proposed_variant}"
    if not trusted_variant.startswith(prefix) or not trusted_variant.endswith(suffix):
        return None
    model_token = trusted_variant[len(prefix) : -len(suffix)]
    if not model_token:
        return None
    return proposed_variant


def validate_evidence(
    evidence: Mapping[str, Any],
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    required_claims: Iterable[str],
    *,
    cas_root: Union[str, Path, None] = None,
    require_family_grounding: bool = False,
    declared_absences: Optional[Mapping[str, Sequence[str]]] = None,
) -> EvidenceValidationReport:
    """Validate verbatim excerpts and exhaustive claim-category coverage.

    Coverage is nominal: a claim category is covered when an excerpt names it, or when
    the proposal declares a typed absence state for it. Whether an excerpt's *text*
    entails the claim is the Codex accuracy checker's judgment, not this function's.

    Claim-category strings are matched EXACTLY. A leaf path ``X.y`` never satisfies the
    aggregate ``X``: an excerpt supporting only ``external_metadata.citation.arxiv_id``
    genuinely does not support the citation as a whole, and rolling leaves up into their
    parent would silently launder that partial support into full coverage.

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
    declared_absences:
        Claim categories carrying a typed, structurally validated absence state, mapped
        to the evidence IDs that state cites. An absence state is a positive,
        evidence-carrying assertion that a fact is not there, so it discharges coverage
        for its own claim; every ID it cites must be a real validated excerpt. Absence
        of the record never discharges anything -- a bare empty value still fails.

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

    absence_covered = _absence_coverage(declared_absences, evidence_ids)
    required = {claim for claim in required_claims if claim}
    missing = required - supported - absence_covered
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
    return EvidenceValidationReport(
        len(excerpts), frozenset(supported), family_grounded, absence_covered
    )


def _absence_coverage(
    declared_absences: Optional[Mapping[str, Sequence[str]]],
    known_evidence_ids: AbstractSet[str],
) -> frozenset[str]:
    """Return claim categories discharged by a typed absence state.

    Parameters
    ----------
    declared_absences:
        Claim category to the evidence IDs its typed absence state cites.
    known_evidence_ids:
        Evidence IDs of excerpts that validated in this same pass.

    Returns
    -------
    frozenset[str]
        Claim categories a typed absence state covers.

    Raises
    ------
    EvidenceValidationError
        If a declared absence is malformed or cites an evidence ID that no validated
        excerpt provides. A fabricated corroboration must never buy coverage.
    """

    if not declared_absences:
        return frozenset()
    covered: set[str] = set()
    for claim, cited in declared_absences.items():
        if not isinstance(claim, str) or not claim.strip():
            raise EvidenceValidationError("declared absence claim names must be non-empty strings")
        # A bare string is a Sequence whose iteration yields characters; require a real
        # list/tuple so a single ID never silently decomposes into per-character IDs.
        if isinstance(cited, (str, bytes)) or not isinstance(cited, Sequence):
            raise EvidenceValidationError(
                f"declared absence for {claim} must carry a list of evidence IDs"
            )
        unknown = [
            evidence_id
            for evidence_id in cited
            if not isinstance(evidence_id, str) or evidence_id not in known_evidence_ids
        ]
        if unknown:
            raise EvidenceValidationError(
                f"declared absence for {claim} cites missing or fabricated evidence: "
                f"{sorted(map(str, unknown))}"
            )
        covered.add(claim)
    return frozenset(covered)


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


def fetched_sources_for_checked_links(
    search_report: Mapping[str, Any],
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> tuple[Mapping[str, Any], ...]:
    """Bind every checked search-report link to one fetched CAS source.

    Parameters
    ----------
    search_report:
        Bounded source search whose ``links_checked`` entries claim inspected content.
    source_manifest:
        Exact controlled-fetch manifest for the author session.

    Returns
    -------
    tuple[Mapping[str, Any], ...]
        Manifest rows in checked-link order, ready for deterministic inventory.

    Raises
    ------
    EvidenceValidationError
        If a checked link was withheld from controlled fetch, is ambiguous, or did
        not produce an inspectable fetched/already-present CAS object.
    """

    links = search_report.get("links_checked")
    if (
        not isinstance(links, list)
        or not links
        or not all(_checked_link_url(link) is not None for link in links)
    ):
        raise EvidenceValidationError("search_report.links_checked must be non-empty URLs")
    link_urls = tuple(str(_checked_link_url(link)) for link in links)
    if len(link_urls) != len(set(link_urls)):
        raise EvidenceValidationError("search_report.links_checked contains duplicate URLs")

    sources_by_url: dict[str, list[Mapping[str, Any]]] = {}
    for source in _source_index(source_manifest).values():
        url = source.get("url")
        if isinstance(url, str) and url:
            sources_by_url.setdefault(url, []).append(source)
    receipts_by_url = _broker_receipts_by_url(source_manifest)

    bound: list[Mapping[str, Any]] = []
    for link in links:
        if isinstance(link, Mapping):
            _validate_policy_checked_link(link, receipts_by_url)
            continue
        assert isinstance(link, str)
        matches = sources_by_url.get(link, [])
        if len(matches) != 1:
            raise EvidenceValidationError(
                f"checked search link must bind to exactly one controlled-fetch source: {link}"
            )
        source = matches[0]
        if source.get("retrieval_status") not in {"fetched", "already-present"}:
            raise EvidenceValidationError(
                f"checked search link has no fetched CAS inventory: {link}"
            )
        if not isinstance(source.get("content_sha256"), str):
            raise EvidenceValidationError(
                f"checked search link has no hash-bound fetched bytes: {link}"
            )
        bound.append(source)
    return tuple(bound)


def _checked_link_url(link: object) -> Optional[str]:
    """Return the URL named by one ``links_checked`` entry, if well-shaped.

    Parameters
    ----------
    link:
        Raw checked-link entry from an author search report.

    Returns
    -------
    str | None
        Non-empty checked URL, or ``None`` when the entry is structurally invalid.
    """

    if isinstance(link, str) and link.strip():
        return link
    if not isinstance(link, Mapping):
        return None
    if set(link) != {"url", "disposition"}:
        return None
    url = link.get("url")
    disposition = link.get("disposition")
    if (
        isinstance(url, str)
        and url.strip()
        and isinstance(disposition, str)
        and disposition in POLICY_CHECKED_LINK_DISPOSITIONS
    ):
        return url
    return None


def _broker_receipts_by_url(
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Index broker outcome receipts by their requested or final URL.

    Parameters
    ----------
    source_manifest:
        Manifest wrapper that may carry ``broker.outcomes``.

    Returns
    -------
    dict[str, list[Mapping[str, Any]]]
        Broker outcome rows keyed by URL.
    """

    if not isinstance(source_manifest, Mapping):
        return {}
    broker = source_manifest.get("broker")
    outcomes = broker.get("outcomes") if isinstance(broker, Mapping) else None
    if not isinstance(outcomes, list):
        return {}
    indexed: dict[str, list[Mapping[str, Any]]] = {}
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            continue
        for field in ("url", "final_url"):
            url = outcome.get(field)
            if isinstance(url, str) and url:
                indexed.setdefault(url, []).append(outcome)
    return indexed


def _validate_policy_checked_link(
    link: Mapping[str, Any],
    receipts_by_url: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Require a policy-typed checked link to be backed by a broker receipt.

    Parameters
    ----------
    link:
        Structured ``links_checked`` entry.
    receipts_by_url:
        Broker outcomes indexed by URL.

    Raises
    ------
    EvidenceValidationError
        If no matching broker receipt proves the policy refusal.
    """

    url = str(link["url"])
    disposition = str(link["disposition"])
    if disposition not in POLICY_CHECKED_LINK_DISPOSITIONS:
        raise EvidenceValidationError(f"checked search link disposition is unsupported: {url}")
    matches = [
        receipt
        for receipt in receipts_by_url.get(url, ())
        if receipt.get("outcome") in POLICY_CHECKED_LINK_OUTCOMES
    ]
    if len(matches) != 1:
        raise EvidenceValidationError(
            "checked search link policy disposition lacks a matching broker receipt: "
            f"{url}"
        )


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
        if start < 0 or end < start or not _same_text(content[start:end], text):
            raise EvidenceValidationError(f"{evidence_id} does not exist verbatim at {locator}")
    elif text not in content:
        # The fold erases which whitespace character sits inside the excerpt; this
        # strip erases whether whitespace sits AROUND it. Line-based extraction
        # tooling appends a trailing newline to whatever it prints (``sed -n 'Np'``
        # ends every emission with one), so an author quoting a 500-byte run out of
        # the middle of a single-line page reproduces every visible byte exactly and
        # still carries one invisible trailing ``\n`` the source does not have at
        # that position. That refusal burned the model's one authoring attempt
        # (menagerie campaign ``pilot``, model ``m4066``, Implicit Q-Learning:
        # excerpt ``ev-arxiv-meta``, byte-identical for its whole visible length).
        #
        # Only the OUTER whitespace of the whole excerpt is dropped, after the fold
        # has already run on both sides. Every visible character, its order, and all
        # interior spacing (modulo the fold's existing run-identity rule) still
        # must match, so an excerpt whose quoted content differs from the source
        # anywhere something is displayed is refused exactly as before. An excerpt
        # that strips to nothing quotes nothing and is refused rather than matched
        # against every source trivially.
        folded = _fold_space(text).strip()
        if not folded or folded not in _fold_space(content):
            raise EvidenceValidationError(
                f"{evidence_id} excerpt is not verbatim in its fetched source"
            )


#: One run of whitespace. Python's ``\s`` over ``str`` is Unicode-aware: it covers every
#: character in category ``Zs`` (U+0020, U+00A0, U+2002-U+200A, U+202F, U+3000, ...) plus
#: the ASCII whitespace controls, and deliberately does NOT cover U+200B ZERO WIDTH SPACE,
#: which is a format character (``Cf``) rather than a separator. That exclusion matters: a
#: zero-width character is invisible but occupies no rendered gap, so folding it away would
#: let an excerpt differ from its source somewhere nothing is displayed. Compiled over
#: decoded text rather than raw bytes because the distinction erased is codepoint-level.
_SPACE_RUN_PATTERN = re.compile(r"\s+")


def _fold_space(value: bytes) -> bytes:
    """Canonicalize every run of Unicode whitespace to one ASCII space.

    A fetched page and the excerpt an author transcribes out of it can differ in
    exactly one respect that carries no meaning: *which* space character sits between
    two runs of visible text. ar5iv renders LaTeX inter-author spacing as U+2003 EM
    SPACE and non-breaking gaps as U+00A0, and an author reading that page writes the
    ordinary U+0020 it renders as. Comparing raw bytes then refuses a transcription
    that reproduced 150 characters of markup -- including exact generated ``id``
    attribute values no one could invent -- over a single invisible codepoint, and
    burns the model's one authoring attempt (menagerie campaign ``pilot``, model
    ``m8245``, MetaFormer/PoolFormer, excerpt ``ev-ar5iv-affiliation``).

    The fold is applied identically to both sides and erases *only* the identity of
    whitespace characters. It can never join two tokens, because a whitespace run
    always folds to one space rather than to nothing, and it can never split one,
    because no space is ever introduced. Every visible character, its order, and every
    token boundary therefore survive untouched: an excerpt naming text the source does
    not contain is refused exactly as before. This is the same equivalence
    :func:`menagerie.crawler.proposal._normalize_support_text` already applies through
    its NFKD fold, which maps these separators to U+0020 -- so the raw-byte check was
    the one place in the pipeline that still treated them as distinct.

    Parameters
    ----------
    value:
        Excerpt or source bytes to canonicalize.

    Returns
    -------
    bytes
        UTF-8 bytes with every whitespace run replaced by a single ASCII space.
    """

    # Deliberately NOT a full NFKC/NFKD pass. Those forms also fold ligatures, full-width
    # forms, and compatibility digits, which are visible-character differences an excerpt
    # must still reproduce exactly. Only the whitespace class is canonicalized here.
    decoded = value.decode("utf-8", "surrogateescape")
    return _SPACE_RUN_PATTERN.sub(" ", decoded).encode("utf-8", "surrogateescape")


def _same_text(content: bytes, text: bytes) -> bool:
    """Return whether two byte runs agree exactly or differ only in space characters."""

    return content == text or _fold_space(content) == _fold_space(text)


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
