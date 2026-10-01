"""Bundle member-relation table (S6, grammar v2): typed, closed, OPTIONAL relations.

One optional relation table per :class:`~torchlens.bundle.Bundle`. Absence
means a plain Bundle with unchanged semantics; membership NEVER requires
alignment or relations (the S6 flexibility invariant). The table is DATA,
not derivation: loads never recompute it, but its invariants (R1/R2/R3) are
re-checked on every load and every mutation.

Pattern-of-record: :mod:`torchlens.distributed._ledger` — closed
vocabularies, frozen payload key sets, fail-closed ``from_payload``,
immutable identity-stable finalized views. Parse boundaries raise
``ValueError``; the public Bundle wiring converts into the typed
:class:`~torchlens.errors.BundleRelationError` family (stable codes on
``exc.fields["code"]``, never message text). The one exception is the
evidence byte budget, which raises ``BundleRelationError`` directly so its
distinct code (``bundle_relation_evidence_over_budget``) survives the
conversion sites.

GRAMMAR v2 (the C07X coordinated tlspec-v9 amendment; foldB s4.4 items 1,
2, 7, 8, 9, 10, 11): every kind's param keys split REQUIRED/OPTIONAL (the
optional sets are per-kind CLOSED; an undeclared key still refuses — the
unknown-key check is untouched). ``successor_of`` admits the optional
``evidence`` envelope (``{schema, items[], facts_digest}``), ``carry_mode``,
and ``state_source`` params. Evidence items are GRADED claims over the
closed :data:`RELATION_CLAIM_GRADES` vocabulary with the closed
:data:`RELATION_UNCHECKED_REASONS` menu (no lane may invent an uncontracted
reason string) and a mandatory nullable ``basis`` key (a schema may demand
it non-null; the grammar can express that without a bump). One per-row
evidence canonical-JSON byte budget applies
(:data:`RELATION_EVIDENCE_BUDGET_BYTES`, the sidecar family default).

DIRECTION PIN (item 8): on a ``successor_of`` row, ``from`` is ALWAYS the
LATER member (the successor) and ``to`` the earlier member it succeeds —
"B succeeds A" persists as ``{"kind": "successor_of", "from": "B",
"to": "A"}``. Version-axis relation rows ORDER members; they never license
cross-member parameter VALUE reads (``checkpoint_series_live_params``).

UNKNOWN-KIND LOADING (loader doctrine leg (a), D5): builtin kinds are BARE
names; extension kinds are NAMESPACED (``"<ns>.<name>"``, at least one dot —
the sidecar family-id grammar). A well-formed row of an unknown NAMESPACED
kind loads OPAQUE and DISCLOSED under ``unknown_kinds="opaque"`` (an
:class:`OpaqueRelationRow`: preserved verbatim on re-save, never executed,
graded ``unchecked(kind_unregistered)`` where a grade applies, skipped by
R1 — its endpoints are unreadable without its kind's schema). A BARE
unknown kind keeps refusing in every mode, and ``unknown_kinds="refuse"``
(``tl.load(unknown_relations="refuse")``) restores the strict behavior for
namespaced kinds too. CONSTRUCTION (``Bundle.relate`` / the constructor)
always refuses unknown kinds: opaque rows enter from artifacts only.

Payload spelling: PAIR rows persist their endpoints under the S6 keys
``from``/``to`` (``from`` is a Python keyword, so the dataclass attributes
are ``from_member``/``to_member``); MEMBER rows persist under ``member``.

PERSISTENCE: ``member_relations`` persists plainly in ``bundle.json`` as of
the coordinated tlspec v8 bump (the pre-release registrar retired empty);
loads re-check R1/R2/R3 on every rebuild. The v2 grammar rides tlspec v9
(the C07X amendment): every new slot is optional, so every v8-era payload
remains valid under v2.

Relation rows are ORDERING/TOPOLOGY claims only. A row asserting a version
axis never licenses cross-member parameter VALUE reads: those refuse typed
(``checkpoint_series_live_params``) until every member carries immutable
capture-time parameter evidence (R8(b), future).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..errors.episode import BundleRelationError

__all__ = [
    "MEMBER_ROW_KINDS",
    "PAIR_ROW_KINDS",
    "RELATION_CLAIM_GRADES",
    "RELATION_EVIDENCE_BUDGET_BYTES",
    "RELATION_UNCHECKED_REASONS",
    "RESERVED_EVIDENCE_SCHEMA_IDS",
    "MemberRelationRow",
    "MemberRelationTable",
    "OpaqueRelationRow",
    "RelationKindSpec",
]

#: Closed relation claim-grade vocabulary (C07X item 9; foldB D12 axis 2).
#: Reserved NOW so the D8 composition rule (outcomes gate whether a check
#: RUNS; evidence never adjudicates an outcome) is expressible without a
#: later coordinated bump. Users extend the RELATION vocabulary, never this
#: TRUST vocabulary.
RELATION_CLAIM_GRADES = frozenset({"verified", "consistent", "disclosed", "unchecked", "divergent"})

#: Closed unchecked-reason menu (C07X items 9-10). D15: a teaching refusal
#: must never state something false, and no lane may invent an uncontracted
#: reason string — new reasons enter by contract change only.
#:
#: - ``incomparable_basis``: statistical fingerprints without a matching
#:   ``basis.recipe`` id never condemn (foldB D14).
#: - ``no_param_snapshot``: the producer is present and healthy; the
#:   parameter snapshot was simply never taken (foldB D15).
#: - ``member_outcome_refused``: the cited member's settled outcome refused
#:   the N2 validation entry, so the item's check never ran (foldB D8).
#: - ``schema_unknown``: the evidence envelope's schema id is not known to
#:   this process; items load opaque (loader doctrine leg (b)).
#: - ``kind_unregistered``: the row's namespaced kind is not registered in
#:   this process; the row loads opaque (loader doctrine leg (a)).
RELATION_UNCHECKED_REASONS = frozenset(
    {
        "incomparable_basis",
        "no_param_snapshot",
        "member_outcome_refused",
        "schema_unknown",
        "kind_unregistered",
    }
)

#: Per-row evidence canonical-JSON byte budget (C07X item 7; the sidecar
#: family default). Evidence envelopes are metadata-sized graded claims,
#: never activation stores; crossing the budget refuses typed
#: (``bundle_relation_evidence_over_budget``).
RELATION_EVIDENCE_BUDGET_BYTES = 1_048_576

#: Reserved evidence schema-ID strings (C07X item 11): registrations, not
#: slots. Their payload validators arrive with their owning lanes; loader
#: doctrine leg (b) makes late arrival safe (an unknown schema id inside a
#: well-formed envelope loads opaque, grades unchecked, and is preserved on
#: re-save — including future ``torchlens.*``-authored ids read by earlier
#: builds).
RESERVED_EVIDENCE_SCHEMA_IDS = frozenset({"version_boundary_v1", "turn_boundary_v1"})

#: Exact key set of the successor_of evidence envelope (C07X item 1).
_EVIDENCE_ENVELOPE_KEYS = frozenset({"schema", "items", "facts_digest"})

#: Namespaced-kind grammar: ``<ns>.<name>`` lowercase identifiers with at
#: least one dot — the sidecar family-id grammar, one namespace grammar for
#: both extension doors (foldB D16).
_NAMESPACED_KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


@dataclass(frozen=True)
class RelationKindSpec:
    """One builtin relation kind's REQUIRED/OPTIONAL param-key split (v2).

    Both sets are CLOSED: a key outside their union refuses
    (``bundle_relation_schema_invalid``) exactly as the v1 exact-set check
    did — the required/optional split does not weaken the undeclared-key
    refusal (foldB D4).
    """

    required: frozenset[str]
    optional: frozenset[str] = frozenset()

    @property
    def declared(self) -> frozenset[str]:
        """The union of required and optional param keys."""

        return self.required | self.optional


#: PAIR-row kinds -> required/optional param-key split (closed, v2).
PAIR_ROW_KINDS: dict[str, RelationKindSpec] = {
    "alternative_of": RelationKindSpec(frozenset()),
    "forked_from": RelationKindSpec(frozenset({"at_step"})),
    "successor_of": RelationKindSpec(
        frozenset(),
        frozenset({"evidence", "carry_mode", "state_source"}),
    ),
    "escalates": RelationKindSpec(frozenset({"at_step"})),
}

#: MEMBER-row kinds -> required/optional param-key split (closed, v2).
MEMBER_ROW_KINDS: dict[str, RelationKindSpec] = {
    "episode_member": RelationKindSpec(frozenset({"episode_id", "at_step", "role"})),
}

_EPISODE_ROLES = frozenset({"prefill", "decode"})

# ``to_payload`` emits exactly these keys per shape; an unknown key in a
# loaded payload is a forged or drifted artifact, never silently ignored.
_PAIR_PAYLOAD_KEYS = frozenset({"kind", "from", "to", "params"})
_MEMBER_PAYLOAD_KEYS = frozenset({"kind", "member", "params"})


def _require_member_name(value: Any, *, key: str, kind: str) -> str:
    """Return a required non-empty member-name string, refusing anything else."""

    if not isinstance(value, str) or not value:
        raise ValueError(
            f"bundle-relation row of kind {kind!r} requires a non-empty member "
            f"name for {key!r}, got {value!r}"
        )
    return value


def _validated_params(kind: str, params: Any) -> dict[str, Any]:
    """Return the closed, per-kind validated param mapping for one row.

    Raises
    ------
    ValueError
        On a non-mapping, an undeclared or missing-required param key, or an
        ill-typed param value (the S6 R2 refusal family; callers convert
        into ``bundle_relation_schema_invalid``).
    BundleRelationError
        ``bundle_relation_evidence_over_budget`` when an evidence envelope
        crosses :data:`RELATION_EVIDENCE_BUDGET_BYTES` (raised directly so
        the distinct code survives the conversion sites).
    """

    spec = PAIR_ROW_KINDS[kind] if kind in PAIR_ROW_KINDS else MEMBER_ROW_KINDS[kind]
    if not isinstance(params, Mapping):
        raise ValueError(
            f"bundle-relation row params must be a mapping, got {type(params).__name__}"
        )
    unknown = set(params) - spec.declared
    if unknown:
        raise ValueError(
            f"bundle-relation row of kind {kind!r} carries undeclared param "
            f"keys {sorted(unknown)}; declared keys are {sorted(spec.declared)} "
            f"(required: {sorted(spec.required)})"
        )
    missing = spec.required - set(params)
    if missing:
        raise ValueError(
            f"bundle-relation row of kind {kind!r} is missing required param keys {sorted(missing)}"
        )
    validated: dict[str, Any] = {}
    for key in sorted(spec.declared):
        if key not in params:
            continue
        value = params[key]
        _validate_param_value(kind, key, value)
        validated[key] = value
    return validated


def _validate_param_value(kind: str, key: str, value: Any) -> None:
    """Type/vocabulary check for one declared relation param value."""

    if key == "at_step":
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"bundle-relation param 'at_step' must be a non-negative int, got {value!r}"
            )
    elif key == "episode_id":
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"bundle-relation param 'episode_id' must be a non-empty string, got {value!r}"
            )
    elif key == "role" and value not in _EPISODE_ROLES:
        raise ValueError(
            f"bundle-relation param 'role' is {value!r}, outside the closed "
            f"vocabulary {sorted(_EPISODE_ROLES)}"
        )
    elif key in ("carry_mode", "state_source"):
        # D16 spellings, reserved at amendment time. Values are DECLARED
        # facts whose vocabularies belong to the F-CHAIN/F-WITNESS lanes;
        # the grammar pins the type envelope only.
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"bundle-relation param {key!r} must be a non-empty string, got {value!r}"
            )
    elif key == "evidence":
        _validate_evidence_envelope(kind, value)


def _validate_evidence_envelope(kind: str, value: Any) -> None:
    """Validate one evidence envelope's SHAPE fail-closed (C07X item 1).

    The envelope is ``{schema, items[], facts_digest}`` exactly. Item
    payload SEMANTICS belong to the schema id's owning validator (none are
    registered today; :data:`RESERVED_EVIDENCE_SCHEMA_IDS` are id
    reservations only), so items are validated as GRADED claims — the
    grade/basis/reason grammar — with schema-specific keys preserved opaque
    (loader doctrine leg (b)).
    """

    if not isinstance(value, Mapping):
        raise ValueError(
            f"bundle-relation 'evidence' must be a mapping envelope, got {type(value).__name__}"
        )
    if set(value) != _EVIDENCE_ENVELOPE_KEYS:
        raise ValueError(
            "bundle-relation 'evidence' envelope must carry exactly the keys "
            f"{sorted(_EVIDENCE_ENVELOPE_KEYS)}, got {sorted(value)}"
        )
    schema = value["schema"]
    if not isinstance(schema, str) or not schema:
        raise ValueError("bundle-relation evidence 'schema' must be a non-empty schema-id string")
    facts_digest = value["facts_digest"]
    if not isinstance(facts_digest, str) or not facts_digest:
        raise ValueError("bundle-relation evidence 'facts_digest' must be a non-empty string")
    items = value["items"]
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError("bundle-relation evidence 'items' must be a list of graded claims")
    for index, item in enumerate(items):
        _validate_evidence_item(index, item)
    _enforce_evidence_budget(kind, value)


def _validate_evidence_item(index: int, item: Any) -> None:
    """Validate one graded evidence claim (grade/basis/reason grammar)."""

    if not isinstance(item, Mapping):
        raise ValueError(
            f"bundle-relation evidence item {index} must be a mapping, got {type(item).__name__}"
        )
    grade = item.get("grade")
    if grade not in RELATION_CLAIM_GRADES:
        raise ValueError(
            f"bundle-relation evidence item {index} carries grade {grade!r}, "
            f"outside the closed vocabulary {sorted(RELATION_CLAIM_GRADES)}"
        )
    if "basis" not in item:
        raise ValueError(
            f"bundle-relation evidence item {index} is missing the 'basis' key "
            "(nullable, but always present — a schema may demand it non-null)"
        )
    basis = item["basis"]
    if basis is not None and (not isinstance(basis, str) or not basis):
        raise ValueError(
            f"bundle-relation evidence item {index} 'basis' must be a non-empty string or null"
        )
    if grade == "unchecked":
        reason = item.get("reason")
        if reason not in RELATION_UNCHECKED_REASONS:
            raise ValueError(
                f"bundle-relation evidence item {index} grades 'unchecked' with "
                f"reason {reason!r}, outside the contracted menu "
                f"{sorted(RELATION_UNCHECKED_REASONS)} (no lane may invent an "
                "uncontracted reason string)"
            )


def _enforce_evidence_budget(kind: str, envelope: Mapping[str, Any]) -> None:
    """Refuse an evidence envelope crossing the canonical-JSON byte budget."""

    try:
        encoded = json.dumps(dict(envelope), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"bundle-relation 'evidence' on kind {kind!r} is not JSON-portable: {exc}. "
            "Evidence envelopes persist inside bundle.json, so payloads are JSON "
            "data (store digests, never tensors)."
        ) from exc
    size = len(encoded.encode("utf-8"))
    if size > RELATION_EVIDENCE_BUDGET_BYTES:
        raise BundleRelationError(
            f"bundle-relation 'evidence' on kind {kind!r} is {size} canonical-JSON "
            f"bytes, above the {RELATION_EVIDENCE_BUDGET_BYTES}-byte per-row budget. "
            "Evidence envelopes are metadata-sized graded claims; move bulk "
            "payloads to a sidecar family and cite them by digest.",
            code="bundle_relation_evidence_over_budget",
            row_kind=kind,
            size_bytes=size,
            budget_bytes=RELATION_EVIDENCE_BUDGET_BYTES,
        )


@dataclass(frozen=True)
class MemberRelationRow:
    """One frozen S6 relation row; ``kind`` selects one of TWO closed shapes.

    PAIR rows (kinds ``alternative_of`` / ``forked_from`` / ``successor_of``
    / ``escalates``) carry ``from_member``/``to_member`` (persisted as the
    S6 ``from``/``to`` keys) and leave ``member`` unset. MEMBER rows (kind
    ``episode_member``) carry ``member`` and leave the endpoints unset.
    ``params`` holds only keys the kind declares — per-kind closed
    REQUIRED/OPTIONAL sets (grammar v2), validated at construction and again
    at every load.

    DIRECTION PIN (C07X item 8): on a ``successor_of`` row ``from_member``
    is ALWAYS the LATER member (the successor) and ``to_member`` the earlier
    member it succeeds — "B succeeds A" is
    ``MemberRelationRow(kind="successor_of", from_member="B", to_member="A")``.
    """

    kind: str
    from_member: str | None = None
    to_member: str | None = None
    member: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind in PAIR_ROW_KINDS:
            _require_member_name(self.from_member, key="from", kind=self.kind)
            _require_member_name(self.to_member, key="to", kind=self.kind)
            if self.member is not None:
                raise ValueError(
                    f"bundle-relation PAIR row of kind {self.kind!r} must not "
                    "carry the MEMBER-shape 'member' endpoint"
                )
        elif self.kind in MEMBER_ROW_KINDS:
            _require_member_name(self.member, key="member", kind=self.kind)
            if self.from_member is not None or self.to_member is not None:
                raise ValueError(
                    f"bundle-relation MEMBER row of kind {self.kind!r} must not "
                    "carry the PAIR-shape 'from'/'to' endpoints"
                )
        else:
            known = sorted(set(PAIR_ROW_KINDS) | set(MEMBER_ROW_KINDS))
            raise ValueError(
                f"bundle-relation row kind {self.kind!r} is outside the closed vocabulary {known}"
            )
        object.__setattr__(self, "params", _validated_params(self.kind, self.params))

    def named_members(self) -> tuple[str, ...]:
        """Return the member names this row references, in schema order."""

        if self.kind in PAIR_ROW_KINDS:
            # __post_init__ proved both endpoints are non-empty strings.
            return (str(self.from_member), str(self.to_member))
        return (str(self.member),)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload in the S6 key spelling."""

        if self.kind in PAIR_ROW_KINDS:
            return {
                "kind": self.kind,
                "from": self.from_member,
                "to": self.to_member,
                "params": dict(self.params),
            }
        return {
            "kind": self.kind,
            "member": self.member,
            "params": dict(self.params),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MemberRelationRow:
        """Rebuild one row from :meth:`to_payload` output, FAIL-CLOSED.

        Raises
        ------
        ValueError
            On an unknown kind, a wrong row shape for the kind, an unknown
            or missing payload/param key, or an ill-typed value (callers
            convert into ``bundle_relation_schema_invalid``).
        BundleRelationError
            ``bundle_relation_evidence_over_budget`` (rides through
            conversion sites unchanged).
        TypeError
            When ``payload`` is not a mapping.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("bundle-relation row payload must be a mapping")
        kind = payload.get("kind")
        if kind in PAIR_ROW_KINDS:
            expected_keys = _PAIR_PAYLOAD_KEYS
        elif kind in MEMBER_ROW_KINDS:
            expected_keys = _MEMBER_PAYLOAD_KEYS
        else:
            known = sorted(set(PAIR_ROW_KINDS) | set(MEMBER_ROW_KINDS))
            raise ValueError(
                f"bundle-relation row kind {kind!r} is outside the closed vocabulary {known}"
            )
        if set(payload) != expected_keys:
            raise ValueError(
                f"bundle-relation row of kind {kind!r} must carry exactly the "
                f"keys {sorted(expected_keys)}, got {sorted(payload)}"
            )
        if kind in PAIR_ROW_KINDS:
            return cls(
                kind=str(kind),
                from_member=_require_member_name(payload["from"], key="from", kind=str(kind)),
                to_member=_require_member_name(payload["to"], key="to", kind=str(kind)),
                params=_validated_params(str(kind), payload["params"]),
            )
        return cls(
            kind=str(kind),
            member=_require_member_name(payload["member"], key="member", kind=str(kind)),
            params=_validated_params(str(kind), payload["params"]),
        )


@dataclass(frozen=True)
class OpaqueRelationRow:
    """One preserved-and-disclosed relation row of an unknown NAMESPACED kind.

    Loader doctrine leg (a) (C07X item 2, foldB D5): a well-formed row whose
    namespaced kind is not registered in this process loads OPAQUE — the raw
    payload is preserved verbatim for byte-exact re-save, the row is
    DISCLOSED (one load warning; :attr:`MemberRelationTable.opaque_kinds`),
    it is never executed, any grade it implies reads
    ``unchecked(kind_unregistered)``, and R1 endpoint validation skips it
    (its endpoints are unreadable without its kind's schema). Opaque rows
    enter from ARTIFACTS only; construction of new rows always refuses
    unknown kinds.
    """

    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _NAMESPACED_KIND_PATTERN.match(self.kind or ""):
            raise ValueError(
                f"opaque relation rows carry NAMESPACED kinds ('<ns>.<name>'); "
                f"got {self.kind!r} (bare unknown kinds refuse, never load opaque)"
            )
        if not isinstance(self.payload, Mapping):
            raise TypeError("opaque relation row payload must be a mapping")
        if self.payload.get("kind") != self.kind:
            raise ValueError(
                f"opaque relation row payload 'kind' {self.payload.get('kind')!r} "
                f"does not match the row kind {self.kind!r}"
            )
        object.__setattr__(self, "payload", dict(self.payload))

    def named_members(self) -> tuple[str, ...]:
        """Opaque rows expose no readable endpoints (R1 skips them)."""

        return ()

    def to_payload(self) -> dict[str, Any]:
        """Return the preserved raw payload (verbatim re-save)."""

        return dict(self.payload)


#: How ``MemberRelationTable.from_payload`` treats unknown NAMESPACED kinds.
UnknownKindPolicy = Literal["opaque", "refuse"]


class MemberRelationTable:
    """Immutable S6 relation table: an identity-stable tuple of frozen rows.

    Mutation never happens in place (R4): ``Bundle.relate`` and the R5
    cascade paths build a NEW validated table, so views handed out never
    change underfoot. Rows are :class:`MemberRelationRow` (validated closed
    grammar) or :class:`OpaqueRelationRow` (preserved unknown namespaced
    kinds, artifacts only).
    """

    def __init__(self, rows: Sequence[MemberRelationRow | OpaqueRelationRow] = ()) -> None:
        for row in rows:
            if not isinstance(row, (MemberRelationRow, OpaqueRelationRow)):
                raise TypeError(
                    "MemberRelationTable rows must be MemberRelationRow or "
                    f"OpaqueRelationRow instances, got {type(row).__name__}"
                )
        self._rows: tuple[MemberRelationRow | OpaqueRelationRow, ...] = tuple(rows)

    @property
    def rows(self) -> tuple[MemberRelationRow | OpaqueRelationRow, ...]:
        """Immutable, identity-stable view of the relation rows in order."""

        return self._rows

    @property
    def opaque_kinds(self) -> tuple[str, ...]:
        """The distinct unknown namespaced kinds preserved opaque, in order."""

        seen: list[str] = []
        for row in self._rows:
            if isinstance(row, OpaqueRelationRow) and row.kind not in seen:
                seen.append(row.kind)
        return tuple(seen)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Any:
        return iter(self._rows)

    def rows_naming(self, member_name: str) -> tuple[MemberRelationRow | OpaqueRelationRow, ...]:
        """Return the rows that reference ``member_name`` (R5 mutator guards)."""

        return tuple(row for row in self._rows if member_name in row.named_members())

    def to_payload(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable payload for portable artifacts.

        Opaque rows re-emit their preserved raw payloads verbatim, in their
        original positions (loader doctrine: preserved on re-save, never
        silently destroyed).
        """

        return [row.to_payload() for row in self._rows]

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        unknown_kinds: UnknownKindPolicy = "refuse",
    ) -> MemberRelationTable:
        """Rebuild a table from :meth:`to_payload` output, FAIL-CLOSED.

        Parameters
        ----------
        payload:
            The list-of-row-payloads value :meth:`to_payload` produced (or
            the equivalent artifact section); anything else refuses.
        unknown_kinds:
            ``"refuse"`` (default; construction semantics — every unknown
            kind refuses) or ``"opaque"`` (loader doctrine leg (a): unknown
            NAMESPACED kinds load as :class:`OpaqueRelationRow`; bare
            unknown kinds still refuse). ``tl.load(unknown_relations=...)``
            selects the policy at the artifact door, defaulting to
            ``"opaque"``.

        Raises
        ------
        ValueError
            On a non-list payload or any invalid row payload (callers
            convert into ``bundle_relation_schema_invalid``).
        BundleRelationError
            ``bundle_relation_evidence_over_budget`` (rides through
            unchanged).
        """

        if not isinstance(payload, list):
            raise ValueError(
                f"bundle-relation table payload must be a list, got {type(payload).__name__}"
            )
        rows: list[MemberRelationRow | OpaqueRelationRow] = []
        for entry in payload:
            if (
                unknown_kinds == "opaque"
                and isinstance(entry, Mapping)
                and isinstance(entry.get("kind"), str)
                and entry["kind"] not in PAIR_ROW_KINDS
                and entry["kind"] not in MEMBER_ROW_KINDS
                and _NAMESPACED_KIND_PATTERN.match(entry["kind"])
            ):
                rows.append(OpaqueRelationRow(kind=entry["kind"], payload=entry))
                continue
            rows.append(MemberRelationRow.from_payload(entry))
        return cls(rows)

    def validate_against_members(self, member_names: Iterable[str]) -> None:
        """Re-check the load-time invariants against a concrete member set.

        R1: every named member must be a current Bundle member — no dangling
        edges, ever (opaque rows are skipped: their endpoints are unreadable
        without their kind's schema, which is exactly why they are disclosed
        as unchecked). R3 kind-scoped invariants: ``episode_member`` rows
        for one ``episode_id`` carry distinct ``at_step`` values and exactly
        one ``role="prefill"`` row, sitting at ``at_step`` 0.

        Raises
        ------
        BundleRelationError
            ``bundle_relation_member_missing`` for a dangling row (R1) or
            ``bundle_relation_schema_invalid`` for a kind-scoped invariant
            violation (R3).
        """

        names = set(member_names)
        for index, row in enumerate(self._rows):
            for named in row.named_members():
                if named not in names:
                    raise BundleRelationError(
                        f"bundle relation row {index} (kind {row.kind!r}) names "
                        f"member {named!r}, which is not a Bundle member. "
                        "Relation rows may only reference current members (S6 R1).",
                        code="bundle_relation_member_missing",
                        row_index=index,
                        row_kind=row.kind,
                        missing_member=named,
                    )
        episodes: dict[str, list[MemberRelationRow]] = {}
        for row in self._rows:
            if isinstance(row, MemberRelationRow) and row.kind == "episode_member":
                episodes.setdefault(str(row.params["episode_id"]), []).append(row)
        for episode_id, rows in episodes.items():
            steps = [int(row.params["at_step"]) for row in rows]
            if len(steps) != len(set(steps)):
                raise BundleRelationError(
                    f"episode_member rows for episode {episode_id!r} carry "
                    "duplicate at_step values; steps must be distinct (S6 R3).",
                    code="bundle_relation_schema_invalid",
                    episode_id=episode_id,
                )
            prefill_rows = [row for row in rows if row.params["role"] == "prefill"]
            if len(prefill_rows) != 1 or int(prefill_rows[0].params["at_step"]) != 0:
                raise BundleRelationError(
                    f"episode_member rows for episode {episode_id!r} must carry "
                    "exactly one role='prefill' row at at_step 0 (S6 R3).",
                    code="bundle_relation_schema_invalid",
                    episode_id=episode_id,
                )
