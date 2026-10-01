"""Bundle-metadata (``bundle.json``) section + member-relation loading.

The C07X preserve-and-disclose loader doctrine (foldB D5) over the two
extension surfaces of ``bundle.json``: the ``member_relations`` table
(unknown NAMESPACED kinds load opaque under
``tl.load(unknown_relations="opaque")``; bare unknown kinds refuse) and the
top-level section namespace (unknown NAMESPACED sections preserve through
the Bundle carriage and re-emit verbatim on save; bare-unknown sections
refuse typed -- the historical silent load-then-destroy is banned in both
directions). Split from ``_io/bundle.py`` at the C07X amendment (R43 size
discipline); the loader (``_load_unified_bundle_directory``) is the one
consumer.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Literal

from .._errors import InvalidArgumentError
from ..errors import TorchLensWarning
from . import TorchLensIOError

#: Writer-owned bare ``bundle.json`` section keys. ``_tlspec_prerelease`` is
#: the historical switched-era marker validated (and refused when stale) by
#: ``validate_prerelease_state`` in the relations loader below. The four
#: lineage keys are the F03-designed shapes for the C07X split-off join
#: (foldB s4.4 item 12): ``bundle_id`` / ``forked_from_bundle_id`` container
#: identity, ``member_construction`` origin anchors, the hash-chained
#: ``operations`` chronology, and the ``member_effect_tables`` registry —
#: all OPTIONAL (absent on pre-F03 artifacts) and fail-closed validated.
_KNOWN_BUNDLE_SECTION_KEYS = frozenset(
    {
        "members",
        "baseline_name",
        "member_relations",
        "_tlspec_prerelease",
        "bundle_id",
        "forked_from_bundle_id",
        "member_construction",
        "operations",
        "member_effect_tables",
    }
)

#: Namespaced-section grammar: same dotted spelling as sidecar family ids
#: and namespaced relation kinds (one namespace grammar for every extension
#: door, foldB D16).
_NAMESPACED_SECTION_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


def _partition_bundle_sections(metadata: dict[str, Any]) -> dict[str, Any]:
    """Partition unknown top-level ``bundle.json`` sections (loader leg (c)).

    The C07X preserve-and-disclose doctrine (foldB D5): an unknown section
    under a NAMESPACED (dotted) key loads opaque — preserved for verbatim
    re-save through the Bundle carriage, disclosed with one warning, never
    executed. An unknown BARE key refuses typed: the historical behavior
    (load silently, destroy on re-save) is banned in BOTH directions.

    Returns
    -------
    dict[str, Any]
        The preserved namespaced sections (possibly empty).

    Raises
    ------
    TorchLensIOError
        ``bundle_section_unknown`` for a bare unknown top-level key.
    """

    preserved: dict[str, Any] = {}
    bare_unknown: list[str] = []
    for key in metadata:
        if key in _KNOWN_BUNDLE_SECTION_KEYS:
            continue
        if isinstance(key, str) and _NAMESPACED_SECTION_PATTERN.match(key):
            preserved[key] = metadata[key]
        else:
            bare_unknown.append(str(key))
    if bare_unknown:
        raise TorchLensIOError(
            f"bundle.json carries unknown bare top-level sections {sorted(bare_unknown)}. "
            "Writer-owned sections are bare; extension sections are namespaced "
            "('<ns>.<name>'), which load opaque and re-save verbatim. A bare "
            "unknown section is a drifted or forged artifact and refuses "
            "fail-closed (it would previously have been silently destroyed on "
            "re-save).",
            code="bundle_section_unknown",
            unknown_sections=sorted(bare_unknown),
        )
    if preserved:
        warnings.warn(
            TorchLensWarning(
                "bundle.json carries unknown namespaced sections "
                f"{sorted(preserved)}; they load OPAQUE (preserved on "
                "re-save, never executed) per the preserve-and-disclose "
                "loader doctrine. Remedy: load with the provider that owns "
                "the sections to consume them, or ignore this disclosure",
                code="bundle_unknown_sections_preserved",
                sections=sorted(preserved),
            ),
            stacklevel=2,
        )
    return preserved


def _load_gated_member_relations(
    metadata: dict[str, Any],
    *,
    unknown_relations: Literal["opaque", "refuse"] = "opaque",
) -> tuple[Any, ...] | None:
    """Read the S6 ``member_relations`` key from bundle metadata.

    The key persists plainly as of tlspec v8. Loads still route through the
    one pre-release validation chokepoint
    (:func:`torchlens._io.prerelease.validate_prerelease_state`) so a
    switched-era artifact (marker present, switch inactive) keeps refusing
    typed and a malformed marker refuses even under the switch; a marker-free
    payload validates against the closed S6 row schema. An absent key is
    simply a plain bundle (S6 R7).

    ``unknown_relations`` is the C07X loader-doctrine policy
    (``tl.load(unknown_relations=...)``): under ``"opaque"`` (the default)
    a well-formed row of an unknown NAMESPACED kind loads as a preserved,
    disclosed ``OpaqueRelationRow`` (leg (a)); ``"refuse"`` restores the
    strict refusal. Bare unknown kinds refuse under both policies.

    Returns
    -------
    tuple | None
        Parsed relation rows for the Bundle constructor (which re-checks R1
        against the loaded member names), or ``None`` when the key is absent.

    Raises
    ------
    PreReleaseArtifactError
        Marker present while the switch is inactive, or a malformed marker.
    BundleRelationError
        ``bundle_relation_schema_invalid`` when the payload is outside the
        closed S6 row schema (``bundle_relation_evidence_over_budget`` rides
        through with its own code).
    """

    if unknown_relations not in ("opaque", "refuse"):
        raise InvalidArgumentError(
            f"unknown_relations must be 'opaque' or 'refuse', got {unknown_relations!r}.",
            code="unknown_relations_policy_invalid",
            remedy="Pass unknown_relations='opaque' (preserve-and-disclose) or 'refuse'.",
        )
    from .prerelease import validate_prerelease_state

    validate_prerelease_state(metadata, cls_name="Bundle")
    relations_payload = metadata.get("member_relations")
    if relations_payload is None:
        return None
    from ..bundle._relations import MemberRelationTable
    from ..errors.episode import BundleRelationError

    try:
        table = MemberRelationTable.from_payload(relations_payload, unknown_kinds=unknown_relations)
    except BundleRelationError:
        # Distinct-code refusals (evidence over budget) keep their own code.
        raise
    except (TypeError, ValueError) as exc:
        raise BundleRelationError(
            f"bundle.json 'member_relations' payload is outside the closed S6 schema: {exc}",
            code="bundle_relation_schema_invalid",
        ) from exc
    opaque_kinds = table.opaque_kinds
    if opaque_kinds:
        warnings.warn(
            TorchLensWarning(
                "bundle.json member_relations carries rows of unknown "
                f"namespaced kinds {list(opaque_kinds)}; they load OPAQUE "
                "(preserved on re-save, never executed, unchecked where a "
                "grade applies) per the preserve-and-disclose loader "
                "doctrine. Remedy: load with the provider that registers "
                "these kinds, or pass tl.load(..., "
                "unknown_relations='refuse') for the strict refusal",
                code="bundle_relation_unknown_kinds_opaque",
                kinds=list(opaque_kinds),
            ),
            stacklevel=2,
        )
    return table.rows


def _load_operation_rows(operations_payload: Any) -> list[Any]:
    """Validate the ``operations`` chronology rows (hash chain rederived)."""

    from ..bundle._lineage import BundleOperation, validate_operation_chain

    if not isinstance(operations_payload, list):
        raise ValueError("bundle.json 'operations' must be a list of rows")
    rows = [BundleOperation.from_payload(row) for row in operations_payload]
    validate_operation_chain(rows)
    return rows


def _load_effect_tables(tables_payload: Any) -> dict[str, Any]:
    """Validate the ``member_effect_tables`` mapping (keys match table ids)."""

    from ..bundle._lineage import MemberEffectTable

    if not isinstance(tables_payload, dict):
        raise ValueError("bundle.json 'member_effect_tables' must map operation_id to table")
    tables: dict[str, Any] = {}
    for operation_id, table_payload in tables_payload.items():
        table = MemberEffectTable.from_payload(table_payload)
        if table.operation_id != operation_id:
            raise ValueError(
                f"member_effect_tables key {operation_id!r} does not match "
                f"its table's operation_id {table.operation_id!r}"
            )
        tables[str(operation_id)] = table
    return tables


def _load_lineage_sections(
    metadata: dict[str, Any],
    *,
    member_names: list[str],
) -> dict[str, Any]:
    """Read + validate the F03 lineage sections from ``bundle.json``.

    All four surfaces are OPTIONAL (a pre-F03 artifact simply lacks them);
    every present surface validates FAIL-CLOSED through the shape validators
    in :mod:`torchlens.bundle._lineage` (anchors over the closed origin
    vocabulary, operation rows re-deriving their hash chain, effect tables
    keyed by their own operation ids).

    Returns
    -------
    dict
        ``{"bundle_id", "forked_from_bundle_id", "member_construction",
        "operations", "member_effect_tables"}`` with ``None``/empty defaults
        for absent sections.

    Raises
    ------
    BundleRelationError
        ``bundle_lineage_invalid`` on any off-shape lineage payload.
    """

    from ..bundle._lineage import validate_member_construction
    from ..errors.episode import BundleRelationError

    loaded: dict[str, Any] = {
        "bundle_id": None,
        "forked_from_bundle_id": None,
        "member_construction": None,
        "operations": [],
        "member_effect_tables": {},
    }
    try:
        for key in ("bundle_id", "forked_from_bundle_id"):
            value = metadata.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"bundle.json {key!r} must be a non-empty string or absent")
            loaded[key] = value
        anchors_payload = metadata.get("member_construction")
        if anchors_payload is not None:
            loaded["member_construction"] = validate_member_construction(
                anchors_payload, member_names=member_names
            )
        operations_payload = metadata.get("operations")
        if operations_payload is not None:
            loaded["operations"] = _load_operation_rows(operations_payload)
        tables_payload = metadata.get("member_effect_tables")
        if tables_payload is not None:
            loaded["member_effect_tables"] = _load_effect_tables(tables_payload)
    except BundleRelationError:
        # Shape validators already raise the typed coded refusal.
        raise
    except (TypeError, ValueError) as exc:
        raise BundleRelationError(
            f"bundle.json lineage sections are outside the F03 closed shapes: {exc}",
            code="bundle_lineage_invalid",
        ) from exc
    return loaded
