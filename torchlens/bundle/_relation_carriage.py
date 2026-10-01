"""Relation-carriage dynamic methods for ``Bundle`` (C07X, foldB S6).

The budget-preserving dynamic methods that carry the S6 member-relation
table: ``Bundle.relate`` appends validated rows (R4 new-table install) and
``Bundle.derive_episode_status`` folds ``episode_member`` rows into the
derived episode status (a DERIVATION, never Bundle-level settlement).
Split out of ``torchlens/bundle/__init__.py`` beside ``_outcome_fold.py``
to respect its size ceiling (R43 split-never-raise); ``__init__`` re-exports
both names and installs them on the accessor table.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from ._relations import MemberRelationRow

if TYPE_CHECKING:
    from ..capture._episode_ledger import EpisodeFoldResult, EpisodeLedger
    from . import Bundle


def _bundle_relate(self: Bundle, *rows: MemberRelationRow | Mapping[str, Any]) -> Bundle:
    """Append S6 relation rows, installing a NEW validated table (R4).

    Exposed as the budget-preserving dynamic method ``Bundle.relate``.

    Parameters
    ----------
    self:
        Bundle receiving the rows.
    *rows:
        ``MemberRelationRow`` instances or payload mappings.

    Returns
    -------
    Bundle
        This bundle.

    Raises
    ------
    BundleRelationError
        ``bundle_relation_schema_invalid`` for an off-schema row,
        ``bundle_relation_member_missing`` for a row naming a non-member
        (R1). On refusal the existing table is unchanged.
    """

    self._member_relations = self._build_relation_table(self._member_relations.rows, rows)
    return self


def _bundle_derive_episode_status(
    self: Bundle,
    episode_id: str,
    *,
    ledger: EpisodeLedger | None = None,
) -> EpisodeFoldResult:
    """Fold a bundle's episode members into a derived episode status.

    Exposed as the budget-preserving dynamic method
    ``Bundle.derive_episode_status``. This is a DERIVATION, never a settled
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

    from ..capture._episode_ledger import derive_episode_status as _fold_episode_status

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
