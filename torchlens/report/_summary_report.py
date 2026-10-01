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
- ``SummaryReport`` subclasses ``str``. On the F08 rebuilt path the text
  IS the canonical byte-stable ASCII contract (memo 3.9); legacy preset
  spellings keep the historical renderer's text byte-identical through
  the compatibility table.

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
    """The summary text WITH its typed, detached data (items 10 + 12).

    A ``str`` subclass: the string payload is the canonical byte-stable
    ASCII contract (memo 3.9), while ``rows`` / ``totals`` / ``capture`` /
    ``to_dict()`` serve the numbers as data. ``__repr__`` is ``__str__``
    (bare display renders the designed form, sumfam D7). Rebuilt-grammar
    results additionally carry the F08 render payload serving
    ``render``/``print``/``to_pandas``/``to_markdown``/``to_html``/
    ``details`` and the scalar conveniences; the report retains neither
    the model nor the Trace and survives cleanup and GC.
    """

    _rows: tuple[SummaryRow, ...]
    _totals: SummaryTotals
    _capture: CaptureFacts
    _rebuilt: Any

    def __new__(
        cls,
        text: str,
        *,
        rows: tuple[SummaryRow, ...],
        totals: SummaryTotals,
        capture: CaptureFacts,
        rebuilt: Any = None,
    ) -> SummaryReport:
        """Build the report around the rendered text."""

        report = super().__new__(cls, text)
        report._rows = rows
        report._totals = totals
        report._capture = capture
        report._rebuilt = rebuilt
        return report

    def __repr__(self) -> str:
        """Bare display renders the designed table, never a str-repr quote."""

        return str(self)

    # ------------------------------------------------------------------
    # F08 result API (memo item 12). Heavy logic lives in _summary_result.

    def _rebuilt_or_refuse(self, method: str) -> Any:
        """The rebuilt payload, or a typed teaching refusal on legacy paths."""

        if self._rebuilt is None:
            from .._errors import InvalidArgumentError

            raise InvalidArgumentError(
                f"{method}() serves rebuilt-grammar summaries; this report was "
                "rendered by a legacy preset spelling.",
                code="summary_result_legacy",
                remedy="call summary() with the rebuilt grammar (bare call, "
                "level=, view=, depth=, ...)",
            )
        return self._rebuilt

    def render(self, style: str = "ascii") -> str:
        """Re-render from data: 'ascii' (canonical), 'unicode', or 'html'."""

        if style == "ascii":
            return str(self)
        payload = self._rebuilt_or_refuse("render")
        if style == "unicode":
            return str(payload.unicode_text)
        if style == "html":
            from ._summary_result import render_result_html

            return render_result_html(payload)
        from .._errors import InvalidArgumentError

        raise InvalidArgumentError(
            f"render() style must be 'ascii', 'unicode', or 'html'; got {style!r}.",
            code="summary_option_invalid",
            remedy="pass style='ascii' (canonical), 'unicode', or 'html'",
        )

    def print(self, style: str = "auto", file: Any = None) -> None:
        """Print with charset resolved at THIS display boundary (memo 3.9)."""

        import sys

        from ._summary_charset import detect_style

        stream = file if file is not None else sys.stdout
        resolved = detect_style(stream) if style == "auto" else style
        if resolved not in ("ascii", "unicode"):
            from .._errors import InvalidArgumentError

            raise InvalidArgumentError(
                f"print() style must be 'auto', 'ascii', or 'unicode'; got {style!r}.",
                code="summary_option_invalid",
                remedy="pass style='auto' (detected), 'ascii', or 'unicode'",
            )
        if resolved == "unicode" and self._rebuilt is None:
            resolved = "ascii"
        text = str(self) if resolved == "ascii" else str(self._rebuilt.unicode_text)
        print(text, file=stream)

    def details(self) -> str:
        """Capture facts as text (the relocated preamble's result-side face)."""

        from ._summary_result import result_details

        return result_details(self)

    def to_pandas(self, scope: str = "display") -> Any:
        """Typed DataFrame projection ('display' rows or 'all' op grain)."""

        from ._summary_result import result_to_pandas

        return result_to_pandas(self, scope)

    def to_markdown(self) -> str:
        """GitHub-flavored markdown table of the rendered rows."""

        from ._summary_result import result_to_markdown

        return result_to_markdown(self)

    def to_html(self) -> str:
        """The dependency-free escaped HTML fragment."""

        payload = self._rebuilt_or_refuse("to_html")
        from ._summary_result import render_result_html

        return render_result_html(payload)

    # Scalar conveniences (raw ints or None; the raw-numbers pin).

    @property
    def total_params(self) -> int | None:
        """Declared parameter total under torch's own identity rule."""

        return self._totals.params_total

    @property
    def executed_params(self) -> int | None:
        """Parameters that participated in the captured forward."""

        return self._totals.params_executed

    @property
    def unexecuted_params(self) -> int | None:
        """Declared-but-never-ran parameters (A3)."""

        return self._totals.params_unexecuted

    @property
    def trainable_params(self) -> int | None:
        """Trainable parameter total."""

        return self._totals.params_trainable

    @property
    def frozen_params(self) -> int | None:
        """Frozen parameter total."""

        return self._totals.params_frozen

    @property
    def total_flops_forward(self) -> int:
        """Forward FLOPs under the stored fma=2 convention."""

        return self._totals.flops_forward_fma2

    @property
    def total_macs_forward(self) -> int:
        """True multiply-accumulate count (never flops//2)."""

        return self._totals.macs_forward

    @property
    def unknown_flop_ops(self) -> int:
        """Ops whose compute is unknown (totals are lower bounds when > 0)."""

        return int(self._totals.compute_coverage.get("unknown", 0))

    @property
    def capture_status(self) -> str | None:
        """The capture outcome status this report was built from."""

        return self._capture.outcome_status

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
        """Dependency-free escaped HTML table (the F08 renderer when rebuilt)."""

        if self._rebuilt is not None:
            from ._summary_result import render_result_html

            return render_result_html(self._rebuilt)
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

    rows, totals, capture = build_summary_data(trace)
    return SummaryReport(text, rows=rows, totals=totals, capture=capture)


def build_summary_data(
    trace: Trace,
) -> tuple[tuple[SummaryRow, ...], SummaryTotals, CaptureFacts]:
    """The DATA stage alone: typed rows, totals, and capture facts."""

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
    return _summary_rows(trace), totals, capture
