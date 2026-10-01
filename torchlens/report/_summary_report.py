"""The detached immutable summary report layer (C02; summary memo item 10).

The structural deliverable: typed ``SummaryRow`` / ``SummaryTotals`` /
``CaptureFacts`` under a ``SummaryReport(str)`` result, built by a split
builder (data from FactCore -> format -> render) so everything after
renders FROM the data instead of minting numbers in f-strings.

Binding laws carried here:

- RAW-NUMBERS PIN (summary memo 3.8, r4): every numeric field is a plain
  int or None -- ``format(raw_field, ',')`` yields digits; the
  MACs-through-the-FLOPs-formatter disease cannot leak into user format
  strings through a typed report.
- DETACHED: the report retains neither the model nor the Trace and
  survives cleanup and GC (pin: build report; del trace; to_dict() works).
- The rendered TEXT is byte-identical to the historical ``summary()``
  output: ``SummaryReport`` subclasses ``str``, so every existing consumer
  and golden holds; F08's ladder/result API re-renders from the same data.

Spellings DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

#: The five-value evidence enum (summary item 10 / costreport D2). Orthogonal
#: to applicability; a mixed aggregate never upgrades past its weakest member.
EVIDENCE_VALUES: tuple[str, ...] = (
    "measured",
    "formula_exact",
    "estimated",
    "hypothesis",
    "unknown",
)


@dataclass(frozen=True)
class SummaryRow:
    """One typed accounting-unit row (op grain; families coincide there).

    ``exclusive`` and ``subtree`` value families are both present so module
    rollups (F08's ladder) extend the SAME row type; at op grain they
    coincide (costreport D5). Every numeric field is a plain int or None.
    """

    row_id: str
    name: str
    kind: str
    site_key: str | None
    shape: tuple[int, ...] | None
    dtype: str | None
    params_exclusive: int | None
    params_subtree: int | None
    flops_exclusive: int | None
    flops_subtree: int | None
    macs_exclusive: int | None
    evidence: str
    reason: str | None


@dataclass(frozen=True)
class SummaryTotals:
    """Headline totals read from FactCore (never re-summed here)."""

    params_total: int | None
    params_per_path_total: int | None
    params_executed: int | None
    params_unexecuted: int | None
    params_trainable: int | None
    params_frozen: int | None
    flops_forward_fma2: int
    macs_forward: int
    at_capture_bytes: int
    retained_now_bytes: int
    compute_coverage: dict[str, int]


@dataclass(frozen=True)
class CaptureFacts:
    """Capture honesty facts the report carries detached."""

    backend: str | None
    model_class_name: str | None
    outcome_status: str | None
    capture_verified: bool | None
    structure_only: bool
    health_verdict: str
    capture_fingerprint: str
    advisories: int


class SummaryReport(str):
    """The summary text WITH its typed, detached data (item 10).

    A ``str`` subclass: every historical consumer, golden, and equality
    check holds byte-for-byte, while ``rows`` / ``totals`` / ``capture`` /
    ``to_dict()`` serve the numbers as data. ``__repr__`` is ``__str__``
    (bare display renders the designed form, sumfam D7).
    """

    _rows: tuple[SummaryRow, ...]
    _totals: SummaryTotals
    _capture: CaptureFacts

    def __new__(
        cls,
        text: str,
        *,
        rows: tuple[SummaryRow, ...],
        totals: SummaryTotals,
        capture: CaptureFacts,
    ) -> SummaryReport:
        """Build the report around the rendered text."""

        report = super().__new__(cls, text)
        report._rows = rows
        report._totals = totals
        report._capture = capture
        return report

    def __repr__(self) -> str:
        """Bare display renders the designed table, never a str-repr quote."""

        return str(self)

    @property
    def rows(self) -> tuple[SummaryRow, ...]:
        """Typed rows (op grain in the substrate; F08 adds rollups)."""

        return self._rows

    @property
    def totals(self) -> SummaryTotals:
        """Headline totals (plain ints; the raw-numbers pin)."""

        return self._totals

    @property
    def capture(self) -> CaptureFacts:
        """Detached capture honesty facts."""

        return self._capture

    @property
    def sections(self) -> tuple[str, ...]:
        """Stable section IDs of the data payload."""

        return ("capture", "totals", "rows")

    def to_dict(self) -> dict[str, Any]:
        """Bounded JSON-safe projection of the data payload."""

        return {
            "schema": "torchlens.summary_report.v1",
            "capture": dict(self._capture.__dict__),
            "totals": {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in self._totals.__dict__.items()
            },
            "rows": [dict(row.__dict__) for row in self._rows],
        }

    def _repr_html_(self) -> str:
        """Dependency-free escaped HTML table (F08 ships the full renderer)."""

        import html

        head = (
            "<table><thead><tr><th>Row</th><th>Kind</th><th>Shape</th>"
            "<th>Params</th><th>Fwd FLOPs</th></tr></thead><tbody>"
        )
        body = []
        for row in self._rows[:200]:
            shape = "" if row.shape is None else "x".join(str(dim) for dim in row.shape)
            body.append(
                "<tr>"
                f"<td>{html.escape(row.name)}</td>"
                f"<td>{html.escape(row.kind)}</td>"
                f"<td>{html.escape(shape)}</td>"
                f"<td>{'' if row.params_exclusive is None else row.params_exclusive:}</td>"
                f"<td>{'' if row.flops_exclusive is None else row.flops_exclusive:}</td>"
                "</tr>"
            )
        truncated = (
            f"<tr><td colspan='5'>... {len(self._rows) - 200} more rows</td></tr>"
            if len(self._rows) > 200
            else ""
        )
        totals = self._totals
        footer = (
            f"<p>params={totals.params_total} "
            f"fwd_flops(fma2)={totals.flops_forward_fma2} "
            f"macs={totals.macs_forward} "
            f"health={html.escape(self._capture.health_verdict)}</p>"
        )
        return head + "".join(body) + truncated + "</tbody></table>" + footer


def _summary_rows(trace: Trace) -> tuple[SummaryRow, ...]:
    """DATA stage: op-grain typed rows from the canonical compute face."""

    from ._compute_truth import compute_aggregation

    rows: list[SummaryRow] = []
    params_by_label: dict[str, int | None] = {}
    shapes_by_label: dict[str, tuple[int, ...] | None] = {}
    for op in getattr(trace, "layer_list", ()) or ():
        label = str(getattr(op, "layer_label", "?"))
        num_params = getattr(op, "num_params", None)
        params_by_label[label] = None if num_params is None else int(num_params)
        shape = getattr(op, "shape", None)
        shapes_by_label[label] = None if shape is None else tuple(int(dim) for dim in shape)
    for row in compute_aggregation(trace).rows:
        params = params_by_label.get(row.label) if row.kind == "op" else None
        rows.append(
            SummaryRow(
                row_id=row.row_id,
                name=row.label,
                kind=row.kind,
                site_key=row.site_key,
                shape=shapes_by_label.get(row.label),
                dtype=row.dtype,
                params_exclusive=params,
                params_subtree=params,
                flops_exclusive=row.flops_fma2,
                flops_subtree=row.flops_fma2,
                macs_exclusive=row.fma_macs,
                evidence=row.evidence,
                reason=row.reason,
            )
        )
    return tuple(rows)


def build_summary_report(trace: Trace, text: str) -> SummaryReport:
    """Attach the detached typed data to one rendered summary text.

    The builder split (data / format / render): the DATA stage reads
    FactCore; the FORMAT/RENDER stages produced ``text`` (the historical
    renderer, byte-stable); this assembler never re-sums anything.
    """

    from ._factcore import factcore
    from ._health import health_facts

    core = factcore(trace)
    facts = health_facts(trace)
    totals = SummaryTotals(
        params_total=core.params.total,
        params_per_path_total=core.params.per_path_total,
        params_executed=core.params.executed,
        params_unexecuted=core.params.unexecuted,
        params_trainable=core.params.trainable,
        params_frozen=core.params.frozen,
        flops_forward_fma2=int(core.compute.partition_total),
        macs_forward=int(core.compute.macs_total),
        at_capture_bytes=core.memory.at_capture_bytes,
        retained_now_bytes=core.memory.retained_now_bytes,
        compute_coverage={
            "known": core.compute.coverage.known,
            "zero_by_rule": core.compute.coverage.zero_by_rule,
            "not_applicable": core.compute.coverage.not_applicable,
            "unknown": core.compute.coverage.unknown,
        },
    )
    outcome = getattr(trace, "outcome", None)
    capture = CaptureFacts(
        backend=getattr(trace, "backend", None),
        model_class_name=getattr(trace, "model_class_name", None),
        outcome_status=getattr(getattr(outcome, "status", None), "value", None),
        capture_verified=getattr(trace, "capture_verified", None),
        structure_only=bool(getattr(trace, "structure_only", False)),
        health_verdict=facts.verdict,
        capture_fingerprint=core.capture_fingerprint,
        advisories=len(
            (getattr(trace, "annotations", {}) or {}).get("capture_advisories", []) or []
        ),
    )
    return SummaryReport(text, rows=_summary_rows(trace), totals=totals, capture=capture)
