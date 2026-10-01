"""The dated, monotone KNOWN-GAP manifest (oracles D25).

Gate mechanics, settled 3/3 by the panel: NEW deltas block from the
registry's first merge (a set difference cannot flake); the LEGACY backlog is
a dated, enumerated, monotone manifest where

- a live red item WITHOUT a manifest row fails loudly (new delta), and
- a manifest row whose item is now green ALSO fails until the row is deleted
  (stale row) -- the manifest only shrinks.

Rows are data, never prose: each names its predicate, its enumerated key, a
date, and an owner, so "known gap" can never rot into "known and ignored
forever" without a visible, counted, dated ledger.
"""

from __future__ import annotations

import csv
import datetime
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

#: Predicate ids a gap row may cite. Closed: a row citing an unknown
#: predicate is a schema error, not a new category.
KNOWN_PREDICATES = frozenset(
    {
        "surface_classification",
        "cache_key_coverage",
        "default_truth",
        "numeral_census",
        "census_delta",
        # 6b (F36): licenses an EMPTY invariance-witness slot on a declared
        # cache-key dont-care row; burned down by funding the witness.
        "invariance_witness",
    }
)


@dataclass(frozen=True)
class KnownGap:
    """One dated, owned, enumerated gap row.

    Parameters
    ----------
    gap_id:
        Stable row id (``GAP-###``).
    predicate:
        The gate this row licenses, from ``KNOWN_PREDICATES``.
    key:
        The enumerated item the gate would otherwise fail on.
    dated:
        ISO date the row was admitted.
    owner:
        The lane/owner on the hook for burning the row down.
    note:
        One-line human rationale.
    """

    gap_id: str
    predicate: str
    key: str
    dated: str
    owner: str
    note: str


def load_known_gaps(path: Path | None = None) -> tuple[KnownGap, ...]:
    """Load and schema-validate the manifest.

    Parameters
    ----------
    path:
        Manifest path; defaults to the committed ``data/known_gaps.tsv``.

    Returns
    -------
    tuple[KnownGap, ...]
        Validated rows.

    Raises
    ------
    ValueError
        On a duplicate id, an unknown predicate, an unparseable date, or an
        empty owner -- schema errors refuse, they never license anything.
    """

    manifest = path if path is not None else DATA_DIR / "known_gaps.tsv"
    rows: list[KnownGap] = []
    seen: set[str] = set()
    with manifest.open(newline="") as handle:
        for record in csv.DictReader(
            (line for line in handle if not line.startswith("#")), delimiter="\t", restval=""
        ):
            gap = KnownGap(**record)
            if gap.gap_id in seen:
                raise ValueError(f"duplicate known-gap id: {gap.gap_id}")
            seen.add(gap.gap_id)
            if gap.predicate not in KNOWN_PREDICATES:
                raise ValueError(f"{gap.gap_id}: unknown predicate {gap.predicate!r}")
            datetime.date.fromisoformat(gap.dated)
            if not gap.owner:
                raise ValueError(f"{gap.gap_id}: empty owner")
            rows.append(gap)
    return tuple(rows)


def licensed_keys(gaps: tuple[KnownGap, ...], predicate: str) -> frozenset[str]:
    """Return the keys the manifest licenses for one predicate.

    Parameters
    ----------
    gaps:
        Loaded manifest rows.
    predicate:
        The gate asking.

    Returns
    -------
    frozenset[str]
        Licensed keys.
    """

    return frozenset(gap.key for gap in gaps if gap.predicate == predicate)


def partition_monotone(
    live_red: frozenset[str],
    licensed: frozenset[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Apply D25 mechanics to one predicate's live evaluation.

    Parameters
    ----------
    live_red:
        Keys the predicate finds red RIGHT NOW on the live tree.
    licensed:
        Keys the manifest licenses for this predicate.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...]]
        ``(new_unlicensed, stale_rows)``: red keys with no manifest row
        (BLOCK -- new delta), and manifest keys no longer red (BLOCK until
        the row is deleted -- monotone-down).
    """

    return tuple(sorted(live_red - licensed)), tuple(sorted(licensed - live_red))
