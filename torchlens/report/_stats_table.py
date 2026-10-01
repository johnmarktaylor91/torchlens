"""stats_table: per-site payload observations as a FactCore projection (C02).

sumfam item 18 / D18: observations of ONE CAPTURED BATCH -- per-site count,
moments, extrema, zero/NaN/Inf census -- rendered from the sound TensorStats
kernel, with TYPED ROW STATES for unsaved / unsupported / disk-backed
payloads (never a hollow zero row) and an EXPLICIT scan-cost policy. It
reports observations, never threshold judgments: a dead unit in one batch
is never called a dead neuron; audit owns judgments.

The substrate face: F10 adds graph-order polish, ``= parent`` marks (behind
the capture-time distribution_relation evidence), the constructor budget
guard with header echo, and sampled-sort disclosure. Spellings
DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from ..stats._tensor_stats import TensorStats

#: Typed row states (D18): every op gets a row; absence of numbers is a
#: STATE, never a fabricated zero.
STATS_ROW_STATES: tuple[str, ...] = ("ok", "unsaved", "disk_backed", "unsupported")

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


@dataclass(frozen=True)
class StatsTableRow:
    """One per-op observation row.

    ``stats`` is the frozen TensorStats record on ``state == 'ok'`` rows and
    ``None`` otherwise; ``state`` names WHY numbers are absent.
    """

    row_id: str
    label: str
    site_key: str | None
    state: str
    stats: TensorStats | None
    reason: str | None


@dataclass(frozen=True)
class StatsTable:
    """The typed observation table (a FactCore projection, session-only)."""

    rows: tuple[StatsTableRow, ...]
    basis: str
    scan_cost_policy: str
    capture_fingerprint: str

    @property
    def ok_rows(self) -> tuple[StatsTableRow, ...]:
        """Rows carrying numbers."""

        return tuple(row for row in self.rows if row.state == "ok")

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
                    "reason": row.reason,
                }
            )
        frame = pd.DataFrame.from_records(records)
        frame.attrs["basis"] = self.basis
        frame.attrs["scan_cost_policy"] = self.scan_cost_policy
        return frame

    def __repr__(self) -> str:
        """Bounded designed card (never a row dump)."""

        states: dict[str, int] = {}
        for row in self.rows:
            states[row.state] = states.get(row.state, 0) + 1
        state_note = ", ".join(f"{state}={count}" for state, count in sorted(states.items()))
        return f"StatsTable({len(self.rows)} rows: {state_note}; basis: {self.basis})"


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


def build_stats_table(trace: Trace) -> StatsTable:
    """Build the per-op observation table over one finished trace."""

    from ..stats._tensor_stats import tensor_stats
    from ._factcore import capture_fingerprint

    rows: list[StatsTableRow] = []
    for op in getattr(trace, "layer_list", ()) or ():
        label = str(getattr(op, "label", getattr(op, "layer_label", "?")))
        state, reason = _row_state(op)
        stats = None
        if state == "ok":
            payload = op.out
            import torch

            if isinstance(payload, torch.Tensor):
                stats = tensor_stats(payload, identity=label)
                if stats.mean is None and stats.true_count is None and stats.numel > 0:
                    unavailable = stats.mean_evidence.reason
                    if unavailable and "unsupported" in unavailable:
                        state, reason, stats = "unsupported", unavailable, None
            else:
                state, reason = "unsupported", f"non-torch payload ({type(payload).__name__})"
        rows.append(
            StatsTableRow(
                row_id=f"op:{label}",
                label=label,
                site_key=getattr(op, "site_key", None),
                state=state,
                stats=stats,
                reason=reason,
            )
        )
    return StatsTable(
        rows=tuple(rows),
        basis=BASIS_NOTE,
        scan_cost_policy=SCAN_COST_POLICY,
        capture_fingerprint=capture_fingerprint(trace),
    )
