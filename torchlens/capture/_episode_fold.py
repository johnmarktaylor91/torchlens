"""The S6-floor episode fold: member outcomes -> a DERIVED episode status.

`derive_episode_status` folds per-step Bundle member outcomes (plus optional
ledger geometry) into an `EpisodeFoldResult` -- a DERIVATION, never a settled
`CaptureOutcome` (the R06 discipline; Bundle has no outcome authority). Split
from `_episode_ledger.py` at the C07X amendment (R43 size discipline); the
ledger module re-exports both names, and `Bundle.derive_episode_status` is
the public consumer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._episode_ledger import (
    _PROVENANCE_TIERS,
    EpisodeFoldStatus,
    ProvenanceTier,
)

if TYPE_CHECKING:
    from ._episode_ledger import EpisodeLedger

__all__ = ["EpisodeFoldResult", "derive_episode_status"]


@dataclass(frozen=True)
class EpisodeFoldResult:
    """Result of the floor fold (section 2.3): a DERIVATION, never settlement.

    Parameters
    ----------
    status:
        Derived episode-status term (closed vocabulary).
    at_step:
        Step index qualifying the ``*_at_step`` terms; ``None`` otherwise.
    member_status:
        The deciding member's settled ``CaptureStatus`` string, disclosed
        verbatim (``ABORTED_NONFINITE`` keeps its exact shipped spelling).
    member_phase:
        The deciding FAILED member's phase, disclosed verbatim, or
        ``"unattributed"`` when the shipped record carries none.
    provenance_tier:
        Episode tier: the MINIMUM of the members' tiers (S2 mixed row),
        ``exact`` only when every member is exact.
    """

    status: EpisodeFoldStatus
    at_step: int | None = None
    member_status: str | None = None
    member_phase: str | None = None
    provenance_tier: ProvenanceTier = "exact"


def _complete_prefix_length(statuses: Sequence[str]) -> int:
    """Length of the leading run of COMPLETE member statuses."""

    length = 0
    for status in statuses:
        if status != "COMPLETE":
            break
        length += 1
    return length


def _fold_effective_tier(
    member_tiers: Sequence[ProvenanceTier] | None, excluded: frozenset[int]
) -> ProvenanceTier:
    """Validate the surviving member tiers and fold them to their minimum."""

    tiers: list[ProvenanceTier] = []
    if member_tiers is not None:
        tiers = [tier for index, tier in enumerate(member_tiers) if index not in excluded]
        for tier in tiers:
            if tier not in _PROVENANCE_TIERS:
                raise ValueError(f"episode provenance tier {tier!r} is out of vocabulary")
    return "ledger_only" if any(tier == "ledger_only" for tier in tiers) else "exact"


def _fold_terminal_tail(
    status: str,
    at_step: int,
    phase: str | None,
    episode_tier: ProvenanceTier,
) -> EpisodeFoldResult | None:
    """Fold arms 3/5/6: the single trailing non-complete member's verdict."""

    if status == "HALTED":
        return EpisodeFoldResult(
            status="episode_halted_at_step",
            at_step=at_step,
            member_status="HALTED",
            provenance_tier=episode_tier,
        )
    if status == "ABORTED_NONFINITE":
        return EpisodeFoldResult(
            status="episode_aborted_at_step",
            at_step=at_step,
            member_status="ABORTED_NONFINITE",
            provenance_tier=episode_tier,
        )
    if status == "FAILED":
        return EpisodeFoldResult(
            status="episode_failed_at_step",
            at_step=at_step,
            member_status="FAILED",
            member_phase=phase if phase is not None else "unattributed",
            provenance_tier=episode_tier,
        )
    return None


def derive_episode_status(
    member_outcomes: Sequence[tuple[str, str | None]],
    *,
    n_declared: int | None,
    ledger: EpisodeLedger | None,
    escalation_members: frozenset[int] | None = None,
    member_tiers: Sequence[ProvenanceTier] | None = None,
) -> EpisodeFoldResult:
    """Fold member outcomes + ledger geometry into a derived episode status.

    THE FOLD IS A TOTAL FUNCTION over (member ``CaptureStatus`` sequence x
    ledger geometry); arms are evaluated in order, FIRST MATCH WINS, with an
    explicit fail-closed default arm. It never writes a settled
    ``CaptureOutcome`` anywhere — there is no Bundle-level settlement.

    Parameters
    ----------
    member_outcomes:
        Ordered ``(CaptureStatus-string, CapturePhase-string-or-None)`` pairs,
        one per declared step member, prefix order.
    n_declared:
        The episode's declared step count — a LEDGER-ONLY declaration fact.
        ``None`` after a pre-bump round-trip without a re-supplied ledger, in
        which case arms 2 and 4 cannot fire and an all-COMPLETE prefix
        degrades FAIL-CLOSED to ``episode_unknown`` (disclosed truth loss,
        spike section 4).
    ledger:
        The episode ledger when available (in-process floor use), else
        ``None``.
    escalation_members:
        Indices of members that are S6 ``escalates`` targets — excluded from
        the fold domain before evaluation (E-B5: they annotate the episode,
        they are not part of the prefix).
    member_tiers:
        Optional per-member provenance tiers; the episode tier is their
        MINIMUM (``ledger_only`` beats ``exact`` downward).

    Returns
    -------
    EpisodeFoldResult
        The derived status with its qualifying disclosures.
    """

    # Lane F40c: the fold is a BLESSING claim over the whole episode; a
    # measured broken join refuses typed BEFORE any arm fires (trace-verb
    # verdict: "a blessing fold refuse typed"). Unmeasured ledgers keep the
    # shipped fold behavior (the F40a disclosure stance).
    if ledger is not None and ledger.header.step_join is not None:
        from ._episode_join import refuse_broken_join_claim

        refuse_broken_join_claim(ledger.header)

    excluded = escalation_members or frozenset()
    members = [pair for index, pair in enumerate(member_outcomes) if index not in excluded]
    episode_tier = _fold_effective_tier(member_tiers, excluded)

    # Arm 1: any member UNATTESTED or UNKNOWN, or the ledger violates the law.
    if ledger is not None:
        from ._episode_ledger import _check_monotone_prefix_law

        try:
            _check_monotone_prefix_law(ledger.rows)
        except ValueError:
            return EpisodeFoldResult(status="episode_unknown", provenance_tier=episode_tier)
    for status, _phase in members:
        if status.upper() in ("UNATTESTED", "UNKNOWN"):
            return EpisodeFoldResult(status="episode_unknown", provenance_tier=episode_tier)

    statuses = [status.upper() for status, _phase in members]
    complete_prefix = _complete_prefix_length(statuses)
    non_complete_tail = statuses[complete_prefix:]

    # Arm 2: all N declared members COMPLETE (needs the ledger-only N).
    if n_declared is not None and not non_complete_tail and len(statuses) == n_declared:
        return EpisodeFoldResult(status="episode_complete", provenance_tier=episode_tier)

    # Arms 3/5/6: exactly one non-complete member, and it is last.
    if len(non_complete_tail) == 1:
        terminal = _fold_terminal_tail(
            non_complete_tail[0], complete_prefix, members[complete_prefix][1], episode_tier
        )
        if terminal is not None:
            return terminal

    # Arm 4: clean COMPLETE prefix + a ledger-declared driver halt at k+1
    # (the until=-at-boundary case: no member for the halted step ever
    # started; by construction witnessed by NO member).
    if (
        not non_complete_tail
        and ledger is not None
        and ledger.truncated_at_step is not None
        and ledger.truncated_at_step == len(statuses)
    ):
        return EpisodeFoldResult(
            status="episode_halted_at_step",
            at_step=len(statuses),
            provenance_tier=episode_tier,
        )

    # Arm 7: anything else — explicit fail-closed default arm. DELIBERATE
    # CASE, not an oversight: a member FAILED post-forward whose driver
    # CONTINUED the episode lands here; episode_unknown is the honest fold.
    return EpisodeFoldResult(status="episode_unknown", provenance_tier=episode_tier)
