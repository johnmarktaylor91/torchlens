"""stats_table: per-site payload observations as a FactCore projection.

sumfam item 18 / D18 (C02 substrate) + the F10 polish (lovely item 9):
observations of ONE CAPTURED BATCH -- per-site count, moments, extrema,
zero/NaN/Inf census -- rendered from the sound TensorStats kernel, with
TYPED ROW STATES for unsaved / unsupported / disk-backed / unscanned
payloads (never a hollow zero row) and an EXPLICIT scan-cost policy. It
reports observations, never threshold judgments: a dead unit in one batch
is never called a dead neuron; audit owns judgments.

F10 additions:
- rows read in GRAPH (execution) order, disclosed on the table;
- ``= parent`` marks render ONLY from the capture-path-stable
  distribution verdict (D28: ``distribution_relation`` derived from the
  closed function-semantics table + persisted geometry; ``view_or_copy``
  and ``data_ptr`` can never authorize a mark); marks never fold by
  default -- ``fold_same_as_parent=True`` is the opt-in flag with an
  exact fold count;
- the total-budget guard lives in the CONSTRUCTOR signature as a declared
  computation-policy request and is echoed once in the header (D29):
  rows past the element budget become typed ``unscanned`` rows, never
  silently absent;
- a table never sorts on a sampled column without saying so in the
  header (``sort_rows`` populates the disclosure).

Spellings DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from ..stats._tensor_stats import TensorStats

#: Typed row states (D18): every op gets a row; absence of numbers is a
#: STATE, never a fabricated zero. ``unscanned`` is the D29 budget state.
STATS_ROW_STATES: tuple[str, ...] = ("ok", "unsaved", "disk_backed", "unsupported", "unscanned")

#: The explicit scan-cost policy line (D4: no surface implicitly triggers a
#: payload scan without saying so; the caller invoked stats_table, which IS
#: the explicit request).
SCAN_COST_POLICY = (
    "stats_table scans every RETAINED payload once through the sound kernel "
    "(exact families per the per-family policy; sampled families marked ~); "
    "disk-backed payloads are NOT materialized"
)

#: The basis disclosure every render must carry (D18): one captured batch.
BASIS_NOTE = "observations of ONE captured batch -- never population claims"

#: Row ordering disclosure (F10): the table reads in execution order.
ORDERING_NOTE = "rows in graph (execution) order"

#: Sortable numeric fields for ``sort_rows`` and their sampled-evidence
#: family on the record (None = always exact).
_SORTABLE_FIELDS: dict[str, str | None] = {
    "mean": "mean_evidence",
    "sd": "sd_evidence",
    "min": None,
    "max": None,
    "zero_count": None,
    "nan_count": None,
    "numel": None,
}


@dataclass(frozen=True)
class StatsTableRow:
    """One per-op observation row.

    ``stats`` is the frozen TensorStats record on ``state == 'ok'`` rows and
    ``None`` otherwise; ``state`` names WHY numbers are absent.
    ``parent_mark`` carries the D28 ``= parent`` disclosure (the parent
    label) with its proof provenance in ``relation_basis`` -- populated
    ONLY from a corroborated capture-time distribution verdict.
    """

    row_id: str
    label: str
    site_key: str | None
    state: str
    stats: TensorStats | None
    reason: str | None
    parent_mark: str | None = None
    relation_basis: str | None = None


@dataclass(frozen=True)
class StatsTable:
    """The typed observation table (a FactCore projection, session-only)."""

    rows: tuple[StatsTableRow, ...]
    basis: str
    scan_cost_policy: str
    capture_fingerprint: str
    ordering: str = ORDERING_NOTE
    budget_note: str | None = None
    sort_disclosure: str | None = None
    fold_note: str | None = None

    @property
    def ok_rows(self) -> tuple[StatsTableRow, ...]:
        """Rows carrying numbers."""

        return tuple(row for row in self.rows if row.state == "ok")

    def sort_rows(self, by: str, *, descending: bool = True) -> StatsTable:
        """Return a re-sorted table; sampled-column sorts are DISCLOSED.

        Parameters
        ----------
        by:
            One of the sortable numeric fields
            (``mean``/``sd``/``min``/``max``/``zero_count``/``nan_count``/
            ``numel``).
        descending:
            Sort direction; rows without numbers keep their relative order
            after all valued rows.

        Returns
        -------
        StatsTable
            New table; ``sort_disclosure`` names the key and, when any
            contributing value was sampled, says so in the header (a table
            never sorts on a sampled column silently).

        Raises
        ------
        ValueError
            Unknown sort field (closed vocabulary; teaching message).
        """

        if by not in _SORTABLE_FIELDS:
            raise ValueError(
                f"stats_table cannot sort on {by!r}; sortable fields are "
                f"{sorted(_SORTABLE_FIELDS)} (closed vocabulary)."
            )

        def value_of(row: StatsTableRow) -> float | None:
            """Extract the sort value from one row, or None."""

            if row.stats is None:
                return None
            if by == "min":
                return row.stats.finite_min
            if by == "max":
                return row.stats.finite_max
            value = getattr(row.stats, by, None)
            return None if value is None else float(value)

        valued = [(value_of(row), index, row) for index, row in enumerate(self.rows)]
        keyed = [entry for entry in valued if entry[0] is not None]
        unvalued = [entry for entry in valued if entry[0] is None]
        keyed.sort(key=lambda entry: (entry[0], entry[1]), reverse=descending)
        ordered = tuple(row for _, _, row in keyed + unvalued)
        evidence_field = _SORTABLE_FIELDS[by]
        sampled = False
        if evidence_field is not None:
            sampled = any(
                row.stats is not None and getattr(row.stats, evidence_field).sampled
                for row in ordered
            )
        disclosure = f"sorted by {by} ({'descending' if descending else 'ascending'})"
        if sampled:
            disclosure += " -- SOME VALUES SAMPLED (~ rows rank by their estimates)"
        return replace(
            self,
            rows=ordered,
            ordering=f"rows re-sorted (was: {self.ordering})",
            sort_disclosure=disclosure,
        )

    def fold_same_as_parent(self) -> StatsTable:
        """Fold ``= parent`` rows out, with an exact count (opt-in only).

        Marks NEVER fold by default (D28); this explicit call removes rows
        whose distribution verdict says they carry the parent's exact value
        multiset, recording the exact fold count on ``fold_note``.
        """

        folded = tuple(row for row in self.rows if row.parent_mark is None)
        count = len(self.rows) - len(folded)
        note = f"{count} '= parent' rows folded (distribution-identical to their parent)"
        return replace(self, rows=folded, fold_note=note)

    def to_pandas(self) -> Any:
        """Optional pandas projection (raw values, evidence preserved)."""

        import pandas as pd

        records = []
        for row in self.rows:
            stats = row.stats
            records.append(
                {
                    "row_id": row.row_id,
                    "label": row.label,
                    "site_key": row.site_key,
                    "state": row.state,
                    "n": None if stats is None else stats.numel,
                    "mean": None if stats is None else stats.mean,
                    "sd": None if stats is None else stats.sd,
                    "min": None if stats is None else stats.finite_min,
                    "max": None if stats is None else stats.finite_max,
                    "zero_count": None if stats is None else stats.zero_count,
                    "nan_count": None if stats is None else stats.nan_count,
                    "inf_count": (
                        None if stats is None else stats.posinf_count + stats.neginf_count
                    ),
                    "sampled": None if stats is None else stats.sd_evidence.sampled,
                    "same_as_parent": row.parent_mark,
                    "relation_basis": row.relation_basis,
                    "reason": row.reason,
                }
            )
        frame = pd.DataFrame.from_records(records)
        frame.attrs["basis"] = self.basis
        frame.attrs["scan_cost_policy"] = self.scan_cost_policy
        frame.attrs["ordering"] = self.ordering
        if self.budget_note:
            frame.attrs["budget"] = self.budget_note
        if self.sort_disclosure:
            frame.attrs["sort_disclosure"] = self.sort_disclosure
        return frame

    def __repr__(self) -> str:
        """Bounded designed card (never a row dump); header echoes policy."""

        states: dict[str, int] = {}
        for row in self.rows:
            states[row.state] = states.get(row.state, 0) + 1
        state_note = ", ".join(f"{state}={count}" for state, count in sorted(states.items()))
        header_extras = [
            note for note in (self.budget_note, self.sort_disclosure, self.fold_note) if note
        ]
        extras = ("; " + "; ".join(header_extras)) if header_extras else ""
        return (
            f"StatsTable({len(self.rows)} rows: {state_note}; {self.ordering}; "
            f"basis: {self.basis}{extras})"
        )


def _row_state(op: Any) -> tuple[str, str | None]:
    """Classify one op's payload availability (typed, never a hollow zero)."""

    from ._agent_json import _payload_state

    state = _payload_state(op)
    if state == "present":
        return "ok", None
    if state == "lazy":
        return "disk_backed", "payload is a lazy blob ref; not materialized by stats_table"
    if getattr(op, "has_saved_activation", False):
        return "unsaved", "saved at capture; no bytes in THIS object"
    return "unsaved", "not retained by the capture-time save= decision"


def _parent_mark(trace: Trace, op: Any) -> tuple[str | None, str | None]:
    """Derive the D28 ``= parent`` mark for one op, fail-closed."""

    from ..intervention.edge_semantics import distribution_relation

    for record in getattr(op, "edge_uses", ()) or ():
        verdict = distribution_relation(trace, record)
        if verdict is not None:
            return (
                f"= {verdict.parent_label}",
                f"{verdict.basis}; {verdict.corroboration}",
            )
    return None, None


def build_stats_table(
    trace: Trace,
    *,
    max_scan_elements: int | None = None,
    mark_same_as_parent: bool = True,
) -> StatsTable:
    """Build the per-op observation table over one finished trace.

    Parameters
    ----------
    trace:
        Finished capture.
    max_scan_elements:
        The D29 total-budget REQUEST: rows are scanned in graph order
        until the cumulative retained element count would exceed this
        budget; later retained rows become typed ``unscanned`` rows and
        the header echoes the budget with the exact skip count. ``None``
        (default) scans everything the scan-cost policy licenses.
    mark_same_as_parent:
        Whether to derive the D28 distribution marks (cheap metadata
        reads; no payload access). Marks never fold by default.

    Returns
    -------
    StatsTable
        Typed observation table in graph order.
    """

    import torch

    from ..stats._tensor_stats import tensor_stats
    from ._factcore import capture_fingerprint

    rows: list[StatsTableRow] = []
    scanned_elements = 0
    skipped_by_budget = 0
    for op in getattr(trace, "layer_list", ()) or ():
        label = str(getattr(op, "label", getattr(op, "layer_label", "?")))
        state, reason = _row_state(op)
        stats = None
        if state == "ok":
            payload = op.out
            if isinstance(payload, torch.Tensor):
                if max_scan_elements is not None and (
                    scanned_elements + payload.numel() > max_scan_elements
                ):
                    state, reason = (
                        "unscanned",
                        f"element budget max_scan_elements={max_scan_elements} exhausted",
                    )
                    skipped_by_budget += 1
                else:
                    scanned_elements += payload.numel()
                    stats = tensor_stats(payload, identity=label)
                    if stats.mean is None and stats.true_count is None and stats.numel > 0:
                        unavailable = stats.mean_evidence.reason
                        if unavailable and "unsupported" in unavailable:
                            state, reason, stats = "unsupported", unavailable, None
            else:
                state, reason = "unsupported", f"non-torch payload ({type(payload).__name__})"
        parent_mark, relation_basis = (None, None)
        if mark_same_as_parent:
            parent_mark, relation_basis = _parent_mark(trace, op)
        rows.append(
            StatsTableRow(
                row_id=f"op:{label}",
                label=label,
                site_key=getattr(op, "site_key", None),
                state=state,
                stats=stats,
                reason=reason,
                parent_mark=parent_mark,
                relation_basis=relation_basis,
            )
        )
    budget_note = None
    if max_scan_elements is not None:
        budget_note = (
            f"budget: max_scan_elements={max_scan_elements} ({skipped_by_budget} rows unscanned)"
        )
    return StatsTable(
        rows=tuple(rows),
        basis=BASIS_NOTE,
        scan_cost_policy=SCAN_COST_POLICY,
        capture_fingerprint=capture_fingerprint(trace),
        budget_note=budget_note,
    )
