"""Bundle lineage: bundle_id, member-construction anchors, BundleOperation rows.

The F03-designed shapes for the C07X split-off join (foldB s4.4 item 12, SOL
r3 proviso applied: the Bundle-side ledger rows waited on F03 to supply their
constructor shape; their ``bundle.json`` carriage — the known-section
partition plus preserve-and-disclose loader — landed with the amendment).
Three persisted surfaces, all OPTIONAL and entry-dark on plain bundles:

- ``bundle_id`` — one random hex identity per container, minted at
  construction, NEVER a hash of timestamps/labels/values (the ledger memo
  D1a id law: content hashes collide across genuinely distinct experiments
  and drift under re-serialization). Forks mint a NEW id and anchor the old
  one; loads restore the saved id verbatim.
- ``member_construction`` — one anchor per member: how this member entered
  THIS container (closed ``origin`` vocabulary), which container/member it
  came from, which BundleOperation created it, and the member trace's
  intervention-event lineage head at entry. Anchors are DISCLOSURE, never
  authority: the trace-side canonical audit remains the construction truth
  the provenance join walks; anchors say where to start walking.
- ``operations`` — the BundleOperation ledger: one row per top-level bundle
  action (``vary`` / ``site_sweep`` / ``fork`` / ...), hash-chained with the
  same ``seq``/``prev_*_digest`` spelling the v9 EVENT audit rows carry, so
  two orderings can never grow separately on one container (foldB brief-delta:
  fork lineage and semantic chronology are SEPARATE typed relations; the
  operation ledger is the chronology).

Effect tables (``MemberEffectTable``) persist beside the operations keyed by
``operation_id`` — the per-candidate effect rows survive member release,
save, load, and an unarmed experiment ledger (ledger memo D3c).

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
Pattern-of-record for the validators: :mod:`torchlens.bundle._relations`
(closed vocabularies, frozen key sets, fail-closed ``from_payload``).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors.episode import BundleRelationError

__all__ = [
    "BUNDLE_OPERATION_KINDS",
    "MEMBER_ORIGIN_VOCABULARY",
    "BundleOperation",
    "MemberEffectRow",
    "MemberEffectTable",
    "mint_bundle_id",
    "validate_member_construction",
]

#: Closed member-origin vocabulary (member_construction anchors).
MEMBER_ORIGIN_VOCABULARY = frozenset(
    {"constructed", "added", "forked", "varied", "swept", "loaded"}
)

#: Closed BundleOperation kind vocabulary. ``do``/``push``/``run`` rows exist
#: so broadcast mutations are chronology too, not only constructions.
BUNDLE_OPERATION_KINDS = frozenset(
    {"construct", "add", "remove", "fork", "do", "push", "run", "vary", "sweep", "site_sweep"}
)

_ANCHOR_KEYS = frozenset(
    {"origin", "source_bundle_id", "source_member", "operation_id", "event_id"}
)
_ANCHOR_REQUIRED = frozenset({"origin"})

_OPERATION_KEYS = frozenset(
    {
        "schema",
        "operation_id",
        "seq",
        "kind",
        "member_names",
        "params",
        "prev_operation_digest",
        "operation_digest",
    }
)

_EFFECT_TABLE_KEYS = frozenset(
    {
        "schema",
        "operation_id",
        "baseline_member",
        "metric_repr",
        "edit_repr",
        "lane",
        "retain_policy",
        "rows",
    }
)

_EFFECT_ROW_KEYS = frozenset(
    {
        "candidate_id",
        "member_name",
        "status",
        "resolved_site_count",
        "value",
        "value_kind",
        "readout_ref",
        "error",
        "retained",
    }
)

#: Closed per-candidate outcome vocabulary (ledger memo D3c: released,
#: refused, and failed candidates are table rows, never falsely members).
_EFFECT_ROW_STATUSES = frozenset({"completed", "released", "refused", "failed"})


def mint_bundle_id() -> str:
    """Mint one random container identity (the D1a id law: never a hash)."""

    return uuid.uuid4().hex[:16]


def validate_member_construction(
    payload: Any, *, member_names: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Validate one persisted ``member_construction`` mapping, FAIL-CLOSED.

    Parameters
    ----------
    payload:
        The ``bundle.json`` section value: ``{member_name: anchor}``.
    member_names:
        Current member names; an anchor naming a non-member refuses (the R1
        no-dangling-edges rule, applied to anchors).

    Returns
    -------
    dict
        Validated anchors (copies).

    Raises
    ------
    BundleRelationError
        ``bundle_lineage_invalid`` on any off-shape payload.
    """

    if not isinstance(payload, Mapping):
        raise BundleRelationError(
            "bundle.json 'member_construction' must be a mapping of member "
            f"name to anchor, got {type(payload).__name__}",
            code="bundle_lineage_invalid",
        )
    names = set(member_names)
    validated: dict[str, dict[str, Any]] = {}
    for name, anchor in payload.items():
        if not isinstance(name, str) or name not in names:
            raise BundleRelationError(
                f"member_construction anchor names {name!r}, which is not a "
                "current Bundle member (anchors may only reference current "
                "members)",
                code="bundle_lineage_invalid",
                anchor_member=str(name),
            )
        if not isinstance(anchor, Mapping):
            raise BundleRelationError(
                f"member_construction anchor for {name!r} must be a mapping, "
                f"got {type(anchor).__name__}",
                code="bundle_lineage_invalid",
                anchor_member=name,
            )
        present = set(anchor)
        if not present >= _ANCHOR_REQUIRED or not present <= _ANCHOR_KEYS:
            raise BundleRelationError(
                f"member_construction anchor for {name!r} carries keys "
                f"{sorted(map(str, present))}; the closed key set is "
                f"{sorted(_ANCHOR_KEYS)} with {sorted(_ANCHOR_REQUIRED)} required",
                code="bundle_lineage_invalid",
                anchor_member=name,
            )
        origin = anchor["origin"]
        if origin not in MEMBER_ORIGIN_VOCABULARY:
            raise BundleRelationError(
                f"member_construction anchor for {name!r} carries origin "
                f"{origin!r}, outside the closed vocabulary "
                f"{sorted(MEMBER_ORIGIN_VOCABULARY)}",
                code="bundle_lineage_invalid",
                anchor_member=name,
            )
        for key in ("source_bundle_id", "source_member", "operation_id", "event_id"):
            value = anchor.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                raise BundleRelationError(
                    f"member_construction anchor for {name!r} key {key!r} must "
                    f"be a non-empty string or null, got {value!r}",
                    code="bundle_lineage_invalid",
                    anchor_member=name,
                )
        validated[name] = dict(anchor)
    return validated


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical-JSON encoding of ``payload``."""

    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BundleOperation:
    """One top-level bundle action: the container-side chronology row.

    ``operation_digest`` covers the row's identity fields plus
    ``prev_operation_digest``, so the persisted list is a hash chain in the
    same spelling as the v9 EVENT audit rows (``seq`` starts at 1; the first
    row's ``prev_operation_digest`` is ``None``). The chain attests
    TorchLens's OWN records only — no harness event ingestion, correlation
    ids only (foldB F03 brief-delta).
    """

    operation_id: str
    seq: int
    kind: str
    member_names: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    prev_operation_digest: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in BUNDLE_OPERATION_KINDS:
            raise ValueError(
                f"BundleOperation kind {self.kind!r} is outside the closed "
                f"vocabulary {sorted(BUNDLE_OPERATION_KINDS)}"
            )
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ValueError("BundleOperation operation_id must be a non-empty string")
        if not isinstance(self.seq, int) or isinstance(self.seq, bool) or self.seq < 1:
            raise ValueError(f"BundleOperation seq must be a positive int, got {self.seq!r}")
        object.__setattr__(self, "member_names", tuple(str(name) for name in self.member_names))
        object.__setattr__(self, "params", dict(self.params))

    @property
    def operation_digest(self) -> str:
        """The chained row digest (identity fields + previous digest)."""

        return _canonical_digest(
            {
                "operation_id": self.operation_id,
                "seq": self.seq,
                "kind": self.kind,
                "member_names": list(self.member_names),
                "params": dict(self.params),
                "prev_operation_digest": self.prev_operation_digest,
            }
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the persisted ``bundle_operation_v1`` row payload."""

        return {
            "schema": "bundle_operation_v1",
            "operation_id": self.operation_id,
            "seq": self.seq,
            "kind": self.kind,
            "member_names": list(self.member_names),
            "params": dict(self.params),
            "prev_operation_digest": self.prev_operation_digest,
            "operation_digest": self.operation_digest,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> BundleOperation:
        """Rebuild one row FAIL-CLOSED, re-deriving and checking the digest."""

        if not isinstance(payload, Mapping):
            raise ValueError("bundle operation row payload must be a mapping")
        if set(payload) != _OPERATION_KEYS:
            raise ValueError(
                "bundle operation row must carry exactly the keys "
                f"{sorted(_OPERATION_KEYS)}, got {sorted(map(str, payload))}"
            )
        if payload["schema"] != "bundle_operation_v1":
            raise ValueError(
                f"bundle operation row schema {payload['schema']!r} is not 'bundle_operation_v1'"
            )
        member_names = payload["member_names"]
        if not isinstance(member_names, Sequence) or isinstance(member_names, (str, bytes)):
            raise ValueError("bundle operation row member_names must be a list")
        params = payload["params"]
        if not isinstance(params, Mapping):
            raise ValueError("bundle operation row params must be a mapping")
        prev_digest = payload["prev_operation_digest"]
        if prev_digest is not None and not isinstance(prev_digest, str):
            raise ValueError("bundle operation row prev_operation_digest must be a string or null")
        row = cls(
            operation_id=str(payload["operation_id"]),
            seq=payload["seq"] if isinstance(payload["seq"], int) else -1,
            kind=str(payload["kind"]),
            member_names=tuple(str(name) for name in member_names),
            params=dict(params),
            prev_operation_digest=prev_digest,
        )
        if row.operation_digest != payload["operation_digest"]:
            raise ValueError(
                f"bundle operation row {row.operation_id!r} digest mismatch: "
                "the persisted operation_digest does not re-derive from the "
                "row's own fields (tampered or drifted artifact)"
            )
        return row


def validate_operation_chain(rows: Sequence[BundleOperation]) -> None:
    """Check the seq/prev-digest chain over an ordered operation list.

    Raises
    ------
    BundleRelationError
        ``bundle_lineage_invalid`` at the first chain break: a seq gap, a
        non-monotone seq, or a ``prev_operation_digest`` that does not equal
        the prior row's digest.
    """

    previous: BundleOperation | None = None
    for row in rows:
        expected_seq = 1 if previous is None else previous.seq + 1
        expected_prev = None if previous is None else previous.operation_digest
        if row.seq != expected_seq or row.prev_operation_digest != expected_prev:
            raise BundleRelationError(
                f"bundle operation chain breaks at seq {row.seq} "
                f"(operation {row.operation_id!r}): expected seq {expected_seq} "
                "with prev_operation_digest equal to the prior row's digest. "
                "The operation ledger is append-only and hash-chained; a break "
                "means rows were dropped, reordered, or edited.",
                code="bundle_lineage_invalid",
                operation_id=row.operation_id,
                seq=row.seq,
            )
        previous = row


@dataclass(frozen=True)
class MemberEffectRow:
    """One candidate's effect-table row (ledger memo D3c).

    Released, refused, and failed candidates keep their rows — the numbers
    survive discard — and ``member_name`` is None exactly when no retained
    member exists for the candidate.
    """

    candidate_id: str
    status: str
    member_name: str | None = None
    resolved_site_count: int | None = None
    value: float | None = None
    value_kind: str = "scalar"
    readout_ref: str | None = None
    error: str | None = None
    retained: bool = False

    def __post_init__(self) -> None:
        if self.status not in _EFFECT_ROW_STATUSES:
            raise ValueError(
                f"effect row status {self.status!r} is outside the closed "
                f"vocabulary {sorted(_EFFECT_ROW_STATUSES)}"
            )
        if self.value_kind not in ("scalar", "opaque"):
            raise ValueError(
                f"effect row value_kind {self.value_kind!r} must be 'scalar' or 'opaque'"
            )
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("effect row candidate_id must be a non-empty string")

    def to_payload(self) -> dict[str, Any]:
        """Return the persisted effect-row payload."""

        return {
            "candidate_id": self.candidate_id,
            "member_name": self.member_name,
            "status": self.status,
            "resolved_site_count": self.resolved_site_count,
            "value": self.value,
            "value_kind": self.value_kind,
            "readout_ref": self.readout_ref,
            "error": self.error,
            "retained": self.retained,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> MemberEffectRow:
        """Rebuild one effect row FAIL-CLOSED."""

        if not isinstance(payload, Mapping) or set(payload) != _EFFECT_ROW_KEYS:
            raise ValueError(
                "effect row payload must carry exactly the keys "
                f"{sorted(_EFFECT_ROW_KEYS)}, got "
                f"{sorted(map(str, payload)) if isinstance(payload, Mapping) else type(payload).__name__}"
            )
        value = payload["value"]
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("effect row value must be a number or null")
        count = payload["resolved_site_count"]
        if count is not None and (not isinstance(count, int) or isinstance(count, bool)):
            raise ValueError("effect row resolved_site_count must be an int or null")
        return cls(
            candidate_id=str(payload["candidate_id"]),
            status=str(payload["status"]),
            member_name=(None if payload["member_name"] is None else str(payload["member_name"])),
            resolved_site_count=count,
            value=None if value is None else float(value),
            value_kind=str(payload["value_kind"]),
            readout_ref=(None if payload["readout_ref"] is None else str(payload["readout_ref"])),
            error=None if payload["error"] is None else str(payload["error"]),
            retained=bool(payload["retained"]),
        )


@dataclass(frozen=True)
class MemberEffectTable:
    """The complete per-candidate effect table for ONE bundle operation.

    Persisted inside the ``.tlspec`` bundle artifact keyed by
    ``operation_id`` (ledger memo D3c): the numbers survive member release,
    save, load, and an unarmed experiment ledger. The table is DATA the
    engine wrote once, never a derivation loads recompute.
    """

    operation_id: str
    rows: tuple[MemberEffectRow, ...] = ()
    baseline_member: str | None = None
    metric_repr: str | None = None
    edit_repr: str | None = None
    lane: str | None = None
    retain_policy: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ValueError("effect table operation_id must be a non-empty string")
        object.__setattr__(self, "rows", tuple(self.rows))
        seen: set[str] = set()
        for row in self.rows:
            if row.candidate_id in seen:
                raise ValueError(
                    f"effect table carries duplicate candidate_id {row.candidate_id!r}"
                )
            seen.add(row.candidate_id)

    def to_payload(self) -> dict[str, Any]:
        """Return the persisted ``member_effect_table_v1`` payload."""

        return {
            "schema": "member_effect_table_v1",
            "operation_id": self.operation_id,
            "baseline_member": self.baseline_member,
            "metric_repr": self.metric_repr,
            "edit_repr": self.edit_repr,
            "lane": self.lane,
            "retain_policy": self.retain_policy,
            "rows": [row.to_payload() for row in self.rows],
        }

    @classmethod
    def from_payload(cls, payload: Any) -> MemberEffectTable:
        """Rebuild one table FAIL-CLOSED."""

        if not isinstance(payload, Mapping) or set(payload) != _EFFECT_TABLE_KEYS:
            raise ValueError(
                "effect table payload must carry exactly the keys "
                f"{sorted(_EFFECT_TABLE_KEYS)}, got "
                f"{sorted(map(str, payload)) if isinstance(payload, Mapping) else type(payload).__name__}"
            )
        if payload["schema"] != "member_effect_table_v1":
            raise ValueError(
                f"effect table schema {payload['schema']!r} is not 'member_effect_table_v1'"
            )
        rows_payload = payload["rows"]
        if not isinstance(rows_payload, Sequence) or isinstance(rows_payload, (str, bytes)):
            raise ValueError("effect table rows must be a list")
        for key in ("baseline_member", "metric_repr", "edit_repr", "lane", "retain_policy"):
            value = payload[key]
            if value is not None and not isinstance(value, str):
                raise ValueError(f"effect table {key} must be a string or null")
        return cls(
            operation_id=str(payload["operation_id"]),
            rows=tuple(MemberEffectRow.from_payload(row) for row in rows_payload),
            baseline_member=payload["baseline_member"],
            metric_repr=payload["metric_repr"],
            edit_repr=payload["edit_repr"],
            lane=payload["lane"],
            retain_policy=payload["retain_policy"],
        )
