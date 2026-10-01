"""Site-key JOIN (guarded, witness-corroborated, verdict-tiered) and the
tier-(a) fold closure -- the S2-INDEPENDENT pure functions of the L1
grouping core.

INTERNAL MODULE: the verdict tiers ride as internal enum values; their
public spellings (and every stamp-consuming surface) wait on the S2
vocabulary amendment. Nothing here is wired into capture defaults --
the join backs the cross-stamp exit-gate machinery and future consumers,
and the closure is entry-dark on plain captures until an affirmative D1
ruling (the flip PR).

THE JOIN RULE (memo 3.1 R1, three layers): the site key is a
STRUCTURAL-POSITION identity -- it proves two ops occupy the same
(module site, type, slot, ordinal) position, never that they originate
from the same source operation. Therefore:

1. PER-CALL-INSTANCE CARDINALITY GUARD: a key joins only when its
   (module_site, layer_type, output_slot) cohort has the SAME
   per-call-instance count MAP on both sides -- strictly stronger than a
   sorted-multiset comparison, which wrongly admits permuted counts
   ((1,3) vs (3,1)) with mis-paired interior ordinals.
2. SOURCE-LOCATION WITNESS: where BOTH sides carry operation witnesses
   (:func:`._site_key.operation_witness` -- the deepest operation frame),
   the witness sets must be equal; disagreement REFUSES the join. This
   catches the equal-cardinality alternative-branch collision that no
   cardinality guard can see.
3. VERDICT TIERS on every joined key: ``corroborated`` (guard + witness
   agree) / ``positional`` (guard passes, witness absent on either side) /
   refused (cardinality or witness). NOTHING is ever verdicted
   "guaranteed" -- same-line re-execution with a count-preserving order
   swap joins as corroborated by design (the disclosed R4 residual:
   "corroborated" means position + call-site provenance agree, never
   "same semantic operation").

TIER-(A) FOLD CLOSURE (memo 5.1, normative): fold group = connected
component of (same recurrent_ops group) UNION (same site_key AND same
equivalence_class). The equivalence side condition is NORMATIVE -- an
unguarded closure mints groups violating the live shared-equivalence
invariant. The closure only coarsens; it never splits a recurrence group,
so every live grouping invariant holds by construction.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .._errors import InvalidArgumentError
from ..data_classes._site_key import ROOT_CALL_INSTANCE, operation_witness, parse_site_key

#: Cohort identity of one structural position family: (module_site,
#: layer_type, output_slot) -- the key minus its ordinal.
_Cohort = tuple[tuple[str, ...], str, int | None]

#: One source-location witness (file, line), or None-parts when unknown.
_Witness = tuple[str | None, int | None]


class _SiteJoinVerdict(str, enum.Enum):
    """Internal join-verdict tiers (public spellings wait on S2)."""

    CORROBORATED = "corroborated"
    POSITIONAL = "positional"
    REFUSED_CARDINALITY = "refused_cardinality"
    REFUSED_WITNESS = "refused_witness"


@dataclass(frozen=True)
class SiteJoinRow:
    """Per-key join outcome, with both guard readings for auditability."""

    verdict: _SiteJoinVerdict
    strong_guard_ok: bool
    weak_guard_ok: bool

    @property
    def joined(self) -> bool:
        """Whether this key joined (corroborated or positional tier)."""

        return self.verdict in (_SiteJoinVerdict.CORROBORATED, _SiteJoinVerdict.POSITIONAL)


@dataclass(frozen=True)
class SiteProfile:
    """Per-capture site structure: keys, cohort cardinalities, witnesses."""

    #: Final op label -> rendered site key.
    keys: dict[str, str]
    #: Cohort -> {pass-qualified call instance -> op count}.
    per_call_counts: dict[_Cohort, dict[str, int]]
    #: Rendered key -> set of operation witnesses (None = witness-absent op).
    witnesses: dict[str, frozenset[_Witness | None]]

    def cohort_of(self, key: str) -> _Cohort:
        """Return the key's cohort identity (the site key minus its ordinal)."""

        module_site, layer_type, output_slot, _ = parse_site_key(key)
        return (module_site, layer_type, output_slot)


def site_profile(trace: Any) -> SiteProfile:
    """Build the join-side profile of one FINISHED capture.

    Reads the persisted per-op facts only (site_key, module_call_stack,
    code_context), so it works identically on live and loaded traces.

    Raises
    ------
    InvalidArgumentError
        ``site_key_unavailable`` when no op carries a site key (legacy
        artifact written before site_key_v1).
    """

    keys: dict[str, str] = {}
    per_call: dict[_Cohort, dict[str, int]] = {}
    witnesses: dict[str, set[_Witness | None]] = {}
    for label in trace.op_labels:
        op = trace.ops[label]
        key = getattr(op, "site_key", None)
        if key is None:
            continue
        keys[label] = key
        module_site, layer_type, output_slot, _ = parse_site_key(key)
        stack = tuple(getattr(op, "module_call_stack", ()) or ())
        call_instance = stack[-1] if stack else ROOT_CALL_INSTANCE
        cohort = (module_site, layer_type, output_slot)
        instance_counts = per_call.setdefault(cohort, {})
        instance_counts[call_instance] = instance_counts.get(call_instance, 0) + 1
        witnesses.setdefault(key, set()).add(operation_witness(op))
    if not keys:
        raise InvalidArgumentError(
            "This trace carries no site keys and cannot join on sites: it "
            "predates the site_key_v1 grouping surface. Site keys are minted "
            "at capture time and their persisted row is pre-release-gated "
            "under tlspec v7, so loaded artifacts read keyless.",
            code="site_key_unavailable",
            remedy=("re-capture the model with a current TorchLens and join on the live traces"),
        )
    return SiteProfile(
        keys=keys,
        per_call_counts=per_call,
        witnesses={key: frozenset(values) for key, values in witnesses.items()},
    )


def _witness_known(values: frozenset[_Witness | None]) -> bool:
    """Whether every op behind a key carries a source-location witness."""

    return bool(values) and None not in values


def join_site_profiles(
    left: SiteProfile,
    right: SiteProfile,
) -> dict[str, SiteJoinRow]:
    """Join two captures' site profiles under the three-layer rule.

    Returns one verdict row per key in the raw key intersection; keys
    present on only one side are honestly unjoined (absent from the
    result). Every returned row carries exactly one verdict tier.
    """

    joined_keys = set(left.keys.values()) & set(right.keys.values())
    rows: dict[str, SiteJoinRow] = {}
    for key in joined_keys:
        cohort = left.cohort_of(key)
        left_counts = left.per_call_counts.get(cohort)
        right_counts = right.per_call_counts.get(cohort)
        strong_ok = left_counts == right_counts
        weak_ok = tuple(sorted((left_counts or {}).values())) == tuple(
            sorted((right_counts or {}).values())
        )
        left_witnesses = left.witnesses.get(key, frozenset({None}))
        right_witnesses = right.witnesses.get(key, frozenset({None}))
        both_known = _witness_known(left_witnesses) and _witness_known(right_witnesses)
        if not strong_ok:
            verdict = _SiteJoinVerdict.REFUSED_CARDINALITY
        elif both_known and left_witnesses != right_witnesses:
            verdict = _SiteJoinVerdict.REFUSED_WITNESS
        elif both_known:
            verdict = _SiteJoinVerdict.CORROBORATED
        else:
            verdict = _SiteJoinVerdict.POSITIONAL
        rows[key] = SiteJoinRow(verdict=verdict, strong_guard_ok=strong_ok, weak_guard_ok=weak_ok)
    return rows


def occurrence_coordinates(trace: Any, keys: dict[str, str]) -> dict[str, tuple[str, str, int]]:
    """Map op label -> (site key, pass-qualified call instance, within-instance position).

    This is exactly the coordinate the per-call-instance cardinality guard
    verifies equal on both sides of a join, so occurrence pairing on it is
    what a ``joined`` verdict licenses — never the label, never a bare
    graph-global ordinal (the leverage panel's measured wrong-row join).
    ``keys`` is a profile's label->key map; keyless ops are absent.
    """

    position: dict[tuple[str, str], int] = {}
    coords: dict[str, tuple[str, str, int]] = {}
    for label in trace.op_labels:
        key = keys.get(label)
        if key is None:
            continue
        stack = tuple(getattr(trace.ops[label], "module_call_stack", ()) or ())
        call_instance = stack[-1] if stack else ROOT_CALL_INSTANCE
        index = position.get((key, call_instance), 0)
        position[(key, call_instance)] = index + 1
        coords[label] = (key, call_instance, index)
    return coords


@dataclass(frozen=True)
class FoldRow:
    """One op's inputs to the tier-(a) closure (policy outputs + facts)."""

    label: str
    site_key: str | None
    equivalence_class: str
    recurrent_labels: tuple[str, ...]


def fold_rows_from_trace(trace: Any) -> list[FoldRow]:
    """Adapter: closure inputs from a finished trace's per-op facts."""

    return [
        FoldRow(
            label=label,
            site_key=getattr(trace.ops[label], "site_key", None),
            equivalence_class=str(trace.ops[label].equivalence_class),
            recurrent_labels=tuple(trace.ops[label].recurrent_ops or ()),
        )
        for label in trace.op_labels
    ]


def fold_site_groups(rows: Iterable[FoldRow]) -> dict[str, frozenset[str]]:
    """Tier-(a) fold closure: label -> its fold group (memo 5.1).

    Union of the existing recurrence relation with (same ``site_key`` AND
    same ``equivalence_class``); keyless rows never fold beyond their
    recurrence group. Pure function; entry-dark on plain captures until an
    affirmative D1 ruling activates a `grouping="fold_sites"` entry path.
    """

    rows = list(rows)
    parent: dict[str, str] = {row.label: row.label for row in rows}

    def find(label: str) -> str:
        """Return the component root of ``label`` with path compression."""

        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(a: str, b: str) -> None:
        """Merge the components containing ``a`` and ``b``."""

        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    _union_recurrence_edges(rows, parent, union)
    _union_guarded_site_edges(rows, union)

    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(find(row.label), []).append(row.label)
    return {label: frozenset(groups[find(label)]) for label in parent}


def _union_recurrence_edges(rows: list[FoldRow], parent: dict[str, str], union: Any) -> None:
    """Union pass 1: the existing recurrence relation (in-roster members)."""

    for row in rows:
        for member in row.recurrent_labels:
            if member in parent:
                union(row.label, member)


def _union_guarded_site_edges(rows: list[FoldRow], union: Any) -> None:
    """Union pass 2: same site_key AND same equivalence_class (the guard)."""

    by_guarded_site: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        if row.site_key is None:
            continue
        by_guarded_site.setdefault((row.site_key, row.equivalence_class), []).append(row.label)
    for members in by_guarded_site.values():
        for member in members[1:]:
            union(members[0], member)
