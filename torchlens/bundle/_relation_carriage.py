"""Relation-carriage dynamic methods for ``Bundle`` (C07X, foldB S6).

The budget-preserving dynamic methods that carry the S6 member-relation
table: ``Bundle.relate`` appends validated rows (R4 new-table install) and
``Bundle.episode_status`` folds ``episode_member`` rows into the
derived episode status (a DERIVATION, never Bundle-level settlement).
Split out of ``torchlens/bundle/__init__.py`` beside ``_outcome_fold.py``
to respect its size ceiling (R43 split-never-raise); ``__init__`` re-exports
both names and installs them on the accessor table.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

from ..errors.episode import BundleRelationError
from ._compare_gate import _ORDERING_RELATION_KINDS
from ._fed_by import FED_BY_KIND, settle_fed_by_params
from ._relations import MEMBER_ROW_KINDS, MemberRelationRow, MemberRelationTable

if TYPE_CHECKING:
    from ..capture._episode_fold import EpisodeFoldResult
    from ..capture._episode_ledger import EpisodeLedger
    from ..data_classes.trace import Trace
    from . import Bundle


def _bundle_relate(
    self: Bundle, *rows: str | MemberRelationRow | Mapping[str, Any], **params: Any
) -> Bundle:
    """Append S6 relation rows, installing a NEW validated table (R4).

    Two spellings. The POSITIONAL form reads as the sentence it records:
    ``b.relate("fed_by", "critique", "draft", token_spans=[[3, 7, 3, 7]])``
    is "critique fed_by draft" (kind, then the LATER member, then the EARLIER
    one; a MEMBER kind such as ``episode_member`` names one member). The ROW
    form passes ``MemberRelationRow`` instances or payload mappings.

    Parameters
    ----------
    self:
        Bundle receiving the rows.
    *rows:
        Either ``(kind, later, earlier)`` / ``(kind, member)`` strings, or
        ``MemberRelationRow`` instances and payload mappings.
    **params:
        The kind's declared params (positional form only). On ``fed_by``
        with ``basis`` omitted, TorchLens binds both members to their token
        streams and verifies the spans when the ids are retained (see
        :mod:`torchlens.bundle._fed_by`); ``token_spans="auto"`` finds the
        spans by exact token-id runs and writes ``basis="matched"`` (see
        :mod:`torchlens.bundle._token_spans`).

    Returns
    -------
    Bundle
        This bundle.

    Raises
    ------
    BundleRelationError
        ``bundle_relation_schema_invalid`` for an off-schema row or a mixed
        spelling, ``bundle_relation_member_missing`` for a row naming a
        non-member (R1), ``bundle_relation_auto_spans_unavailable``,
        ``bundle_relation_auto_spans_ambiguous`` and
        ``bundle_relation_auto_spans_unmatched`` for ``token_spans="auto"``,
        ``bundle_relation_span_out_of_bounds`` and
        ``bundle_relation_span_unverified`` for ``fed_by`` spans. On refusal
        the existing table is unchanged.
    """

    if rows and isinstance(rows[0], str):
        names = cast("tuple[str, ...]", rows)
        row = _positional_row(names, params)
        if params.get("token_spans") == "auto":
            row = {**row, "params": _auto_params(self, row, params)}
        self._member_relations = self._build_relation_table(
            self._member_relations.rows, [row], assess_unclaimed=True
        )
        return self
    if params:
        raise BundleRelationError(
            f"Bundle.relate got keyword params {sorted(params)} with row objects; "
            "keyword params belong to the positional spelling "
            "relate(kind, later, earlier, **params).",
            code="bundle_relation_schema_invalid",
            remedy="put the params inside each row, or use the positional spelling",
        )
    rows_only = cast("tuple[MemberRelationRow | Mapping[str, Any], ...]", rows)
    self._member_relations = self._build_relation_table(self._member_relations.rows, rows_only)
    return self


def _positional_row(names: tuple[str, ...], params: Mapping[str, Any]) -> dict[str, Any]:
    """Return the row payload for ``relate(kind, later, earlier, **params)``."""

    kind, members = names[0], names[1:]
    if not all(isinstance(name, str) for name in members):
        raise BundleRelationError(
            "Bundle.relate positional form takes member NAMES after the kind; "
            "do not mix it with row objects.",
            code="bundle_relation_schema_invalid",
            remedy="pass relate(kind, later, earlier, **params) with member names",
        )
    if params.get("token_spans") == "auto" and (
        kind != FED_BY_KIND or len(members) != 2 or {"basis", "coverage"} & set(params)
    ):
        raise BundleRelationError(
            "token_spans='auto' asks TorchLens to find the spans of one "
            "relate('fed_by', later, earlier) row and to compute its basis and "
            f"coverage; got kind {kind!r} with {len(members)} member name(s) and "
            f"params {sorted(params)}.",
            code="bundle_relation_schema_invalid",
            remedy="use token_spans='auto' only on fed_by, without basis or coverage",
            row_kind=kind,
        )
    if kind in MEMBER_ROW_KINDS and len(members) == 1:
        return {"kind": kind, "member": members[0], "params": dict(params)}
    if len(members) != 2:
        raise BundleRelationError(
            f"Bundle.relate({kind!r}, ...) takes two member names (later, earlier) "
            f"for a pair kind or one for a member kind; got {len(members)}.",
            code="bundle_relation_schema_invalid",
            remedy="name the later member, then the earlier member",
            row_kind=kind,
        )
    return {"kind": kind, "from": members[0], "to": members[1], "params": dict(params)}


def _auto_params(self: Bundle, row: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``params`` with ``token_spans='auto'`` replaced by the matched spans."""

    from ._token_spans import auto_token_spans

    later, earlier = str(row["from"]), str(row["to"])
    for name in (later, earlier):
        if name not in self._members:
            raise BundleRelationError(
                f"Bundle.relate('fed_by', {later!r}, {earlier!r}, token_spans='auto') names "
                f"{name!r}, which is not a Bundle member (S6 R1).",
                code="bundle_relation_member_missing",
                remedy="add the named member before relating it",
                row_kind=FED_BY_KIND,
                missing_member=name,
            )
    spans = auto_token_spans(self._members[later], self._members[earlier])
    return {**params, "token_spans": spans, "basis": "matched"}


def settle_relation_rows(
    table: MemberRelationTable,
    members: Mapping[str, Trace],
    *,
    first_new: int,
    assess_unclaimed: bool,
) -> MemberRelationTable:
    """Return ``table`` with every NEW ``fed_by`` row's basis settled against its members.

    Claimed ``verified``/``matched`` rows re-check (unverifiable clamps to
    ``declared``, contradicted refuses); spans are bounds-checked; the
    positional door also binds and verifies unclaimed rows.
    """

    rows = list(table.rows)
    changed = False
    for index in range(first_new, len(rows)):
        row = rows[index]
        if not isinstance(row, MemberRelationRow) or row.kind != FED_BY_KIND:
            continue
        settled = settle_fed_by_params(
            row.params,
            members[str(row.from_member)],
            members[str(row.to_member)],
            row_index=index,
            assess_unclaimed=assess_unclaimed,
        )
        if settled != dict(row.params):
            rows[index] = MemberRelationRow(
                kind=row.kind, from_member=row.from_member, to_member=row.to_member, params=settled
            )
            changed = True
    return MemberRelationTable(rows) if changed else table


def _ordering_walk(self: Bundle, name: str | Trace, *, upstream: bool) -> tuple[str, ...]:
    """Breadth-first walk over the ordering kinds, nearest member first.

    Raises
    ------
    KeyError
        When ``name`` is not a member (the Bundle's own lookup).
    """

    start = self._coerce_member_name(name)
    if start not in self._members:
        raise KeyError(f"{start!r} is not a member of this Bundle; members: {self.names}")
    adjacency: dict[str, list[str]] = {}
    for row in self._member_relations.rows:
        if isinstance(row, MemberRelationRow) and row.kind in _ORDERING_RELATION_KINDS:
            later, earlier = str(row.from_member), str(row.to_member)
            head, tail = (later, earlier) if upstream else (earlier, later)
            adjacency.setdefault(head, []).append(tail)
    order: list[str] = []
    frontier, seen = [start], {start}
    while frontier:
        current = frontier.pop(0)
        for neighbor in adjacency.get(current, []):
            if neighbor not in seen:
                seen.add(neighbor)
                order.append(neighbor)
                frontier.append(neighbor)
    return tuple(order)


def _bundle_upstream(self: Bundle, name: str | Trace) -> tuple[str, ...]:
    """Return every member ``name`` transitively came after, nearest first.

    Walks the ordering relation kinds (``fed_by``, ``selected_by``,
    ``successor_of``, ``forked_from``, ``escalates``) from the later member to the earlier one:
    for "critique fed_by draft", ``upstream("critique")`` is ``("draft",)``.
    Exposed as the dynamic method ``Bundle.upstream``.
    """

    return _ordering_walk(self, name, upstream=True)


def _bundle_downstream(self: Bundle, name: str | Trace) -> tuple[str, ...]:
    """Return every member that transitively came after ``name``, nearest first.

    The reverse of :func:`_bundle_upstream` over the same ordering rows:
    ``downstream("draft")`` is ``("critique", ...)``. Exposed as the dynamic
    method ``Bundle.downstream``.
    """

    return _ordering_walk(self, name, upstream=False)


def register_relation_walks(table: dict[str, Callable[..., Any]]) -> None:
    """Install the walk helpers on the Bundle dynamic-method table."""

    from ._alternatives import _bundle_add_alternative
    from ._token_divergence import _bundle_token_divergence
    from ._token_spans import register_token_pairs

    table["upstream"] = _bundle_upstream
    table["downstream"] = _bundle_downstream
    register_token_pairs(table)
    # Steer one call and re-run the harness (harness.2).
    table["token_divergence"] = _bundle_token_divergence
    table["add_alternative"] = _bundle_add_alternative


def _bundle_derive_episode_status(
    self: Bundle,
    episode_id: str,
    *,
    ledger: EpisodeLedger | None = None,
) -> EpisodeFoldResult:
    """Fold a bundle's episode members into a derived episode status.

    Exposed as the budget-preserving dynamic method
    ``Bundle.episode_status``. This is a DERIVATION, never a settled
    outcome: the fold recomputes from member outcomes plus optional ledger
    geometry, writes nothing, and there is no Bundle-level settlement
    (Bundle has no outcome FIELD by design; members remain the settlement
    authority, and ``Bundle.outcome`` is likewise only the derived
    worst-of-members disclosure fold).

    Parameters
    ----------
    self:
        Bundle whose episode members are folded.
    episode_id:
        Episode entity named by ``episode_member`` relation rows.
    ledger:
        Optional episode ledger; supplies ``n_steps_declared`` and the
        driver-halt geometry (fold arms 2 and 4 are ledger-only facts and
        degrade fail-closed to ``episode_unknown`` without it).

    Returns
    -------
    EpisodeFoldResult
        The derived status with its qualifying disclosures. An
        ``episode_id`` with no relation rows folds over an empty domain and
        lands on the fail-closed ``episode_unknown`` default arm.
    """

    from ..capture._episode_fold import derive_episode_status as _fold_episode_status

    # ``isinstance`` narrows past OpaqueRelationRow (preserved unknown
    # namespaced kinds carry no readable params/endpoints and never fold).
    episode_rows = sorted(
        (
            row
            for row in self._member_relations.rows
            if isinstance(row, MemberRelationRow)
            and row.kind == "episode_member"
            and row.params["episode_id"] == episode_id
        ),
        key=lambda row: int(row.params["at_step"]),
    )
    escalation_sources = {
        row.from_member
        for row in self._member_relations.rows
        if isinstance(row, MemberRelationRow) and row.kind == "escalates"
    }
    member_outcomes: list[tuple[str, str | None]] = []
    excluded: set[int] = set()
    for index, row in enumerate(episode_rows):
        member = self._members[cast("str", row.member)]
        # Public settled-outcome accessor; None (unsettled live trace)
        # folds as UNKNOWN — the fold's most restrictive input.
        outcome = getattr(member, "outcome", None)
        if outcome is None:
            member_outcomes.append(("unknown", None))
        else:
            phase = outcome.phase.value if outcome.phase is not None else None
            member_outcomes.append((outcome.status.value, phase))
        # E-B5: escalation members annotate the episode; they are not part
        # of the prefix law and leave the fold domain here.
        if row.member in escalation_sources:
            excluded.add(index)
    n_declared = ledger.header.n_steps_declared if ledger is not None else None
    return _fold_episode_status(
        member_outcomes,
        n_declared=n_declared,
        ledger=ledger,
        escalation_members=frozenset(excluded),
    )
