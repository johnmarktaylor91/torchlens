"""The one typed ``tl.debug`` report object (U42.2; spelling DOCUMENTED-UNSTABLE).

Every report-shaped ``tl.debug`` result is a :class:`DebugReport`: the typed
results (``TraceAudit``, ``DeterminismReport``, ``NanReport``, ...) subclass
it, and the tabular helpers (``hot_path``, ``dead_neurons``, ``compare``,
``recompute_candidates``, ``gradient_flow_audit``, ``compare_params``)
return a tabular :class:`DebugReport` whose ``.to_pandas()`` is the frame
they used to return. The object HAS a table, it never IS one (the F10
report protocol, ``torchlens.report._report_protocol.BoundedReport``).

The common surface:

- ``report_type`` / ``subject`` / ``status``: the lead facts;
- ``to_dict()``: a plain, JSON-friendly payload (tensors and graph records
  are summarized, never inlined);
- ``to_pandas()``: the tabular exit;
- ``str(report)``: a bounded card (lead line, at most five findings, the
  exact remainder, the exits). Reports that already print a house-voice
  summary keep it unchanged.

The public spelling is ``torchlens.debug.DebugReport``. This module sits at
the package root (not inside ``torchlens.debug``) so ``torchlens.checks``
can subclass it without importing the ``debug`` package, which itself
imports ``checks``.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping, Sequence
from typing import Any

__tl_layer__ = "L1"

#: Nesting depth past which :meth:`DebugReport.to_dict` summarizes values.
_MAX_PLAIN_DEPTH = 6

#: Exits every card names on its ``More:`` line.
REPORT_EXITS: tuple[str, ...] = (".to_pandas()", ".to_dict()")


def _require_pandas() -> Any:
    """Import pandas with the tabular-extra guidance."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "DebugReport.to_pandas() requires pandas. Install with `pip install torchlens[tabular]`."
        ) from exc
    return pd


#: Sentinel: :func:`_plain_structured` did not recognize a container.
_NOT_STRUCTURED = object()


def _plain(value: Any, depth: int = 0) -> Any:
    """Convert ``value`` into a JSON-friendly payload without inlining payloads.

    Parameters
    ----------
    value:
        Any report field value.
    depth:
        Current nesting depth.

    Returns
    -------
    Any
        Primitives unchanged; containers converted element-wise; tensors as a
        shape/dtype summary; graph records (anything with a ``label``) as
        their label; other objects as ``"<TypeName>"``.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth >= _MAX_PLAIN_DEPTH:
        return f"<{type(value).__name__}>"
    converted = _plain_structured(value, depth)
    return _plain_leaf(value) if converted is _NOT_STRUCTURED else converted


def _plain_structured(value: Any, depth: int) -> Any:
    """Convert a report, enum, mapping, dataclass or sequence element-wise."""

    if isinstance(value, DebugReport):
        return value.to_dict()
    if isinstance(value, enum.Enum):
        return _plain(value.value, depth + 1)
    if isinstance(value, Mapping):
        return {str(key): _plain(item, depth + 1) for key, item in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            spec.name: _plain(getattr(value, spec.name), depth + 1)
            for spec in dataclasses.fields(value)
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return [_plain(item, depth + 1) for item in items]
    return _NOT_STRUCTURED


def _plain_leaf(value: Any) -> str:
    """Summarize a non-container value: dtype/device text, tensor shape, label."""

    if type(value).__name__ in ("dtype", "device", "Size"):
        return str(value)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return f"<{type(value).__name__} shape={tuple(shape)} dtype={dtype}>"
    label = getattr(value, "label", None)
    if isinstance(label, str):
        return label
    return f"<{type(value).__name__}>"


class DebugReport:
    """Base class of every report-shaped ``tl.debug`` result.

    Subclasses state their lead through three hooks: ``_report_subject``
    (what was examined), ``_report_status`` (the verdict token) and
    ``_report_findings`` (one-line findings, most severe first). Dataclass
    subclasses get ``to_dict()`` and ``to_pandas()`` from their fields.
    DOCUMENTED-UNSTABLE spelling.
    """

    __slots__ = ()

    #: Reports whose ``__repr__`` is already the house-voice printed summary
    #: keep printing it; the card is then available as :meth:`card`.
    _prints_repr = False

    #: The subject a report prints when it does not compute one.
    _subject_text = "trace"

    @property
    def report_type(self) -> str:
        """The report's type name (the lead's first token)."""

        return type(self).__name__

    @property
    def subject(self) -> str:
        """What the report examined."""

        return str(self._report_subject())

    @property
    def status(self) -> str:
        """The report's verdict token."""

        return str(self._report_status())

    def _report_subject(self) -> str:
        """Return the subject; subclasses override or set ``_subject_text``."""

        return self._subject_text

    def _report_status(self) -> str:
        """Return the verdict token; subclasses override."""

        return "ok"

    def _report_findings(self) -> list[str]:
        """Return the one-line findings, most severe first; subclasses override."""

        return []

    def _report_coverage(self) -> str | None:
        """Return the coverage statement, if the report has one."""

        return None

    def _report_severity_counts(self) -> dict[str, int]:
        """Return severity-token counts for the lead line."""

        return {}

    def _report_rows(self) -> list[dict[str, Any]] | None:
        """Return the report's natural table rows, or ``None`` for one lead row."""

        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-friendly payload of the report.

        Returns
        -------
        dict[str, Any]
            ``report_type``, ``subject`` and ``status``, then every field of
            a dataclass report (tensors and graph records summarized).
        """

        payload: dict[str, Any] = {
            "report_type": self.report_type,
            "subject": self.subject,
            "status": self.status,
        }
        if dataclasses.is_dataclass(self):
            for spec in dataclasses.fields(self):
                payload.setdefault(spec.name, _plain(getattr(self, spec.name)))
        return payload

    def to_pandas(self) -> Any:
        """Return the tabular projection of the report.

        Returns
        -------
        pandas.DataFrame
            The report's natural rows when it has them (findings, breaks,
            transitions, ...), else one row of its lead and scalar fields.
        """

        pd = _require_pandas()
        rows = self._report_rows()
        if rows is not None:
            return pd.DataFrame(
                [{key: _plain(value) for key, value in row.items()} for row in rows]
            )
        scalar = {
            key: value
            for key, value in self.to_dict().items()
            if value is None or isinstance(value, (bool, int, float, str))
        }
        return pd.DataFrame([scalar])

    def card(self) -> str:
        """Return the bounded card: lead, at most five findings, exact remainder.

        Returns
        -------
        str
            The rendered card.
        """

        from .report._report_protocol import ReportLead, render_report_card

        lead = ReportLead(
            report_type=self.report_type,
            subject=self.subject,
            status=self.status,
            severity_counts=self._report_severity_counts(),
            coverage=self._report_coverage(),
        )
        return render_report_card(lead, self._report_findings(), exits=REPORT_EXITS)

    def __str__(self) -> str:
        """Print the house-voice summary or the bounded card."""

        if self._prints_repr:
            return repr(self)
        return self.card()


class FoundReport(DebugReport):
    """A :class:`DebugReport` whose verdict is a ``found`` flag.

    The status is ``"found"`` / ``"not_found"`` and the lead finding is the
    report's ``message``; subclasses add subject, coverage or more findings.
    """

    __slots__ = ()

    def _report_status(self) -> str:
        """The verdict token on the card's lead line."""

        return "found" if getattr(self, "found", False) else "not_found"

    def _report_findings(self) -> list[str]:
        """One-line findings for the card, most severe first."""

        message = getattr(self, "message", "")
        return [str(message)] if message else []


def rows_of(records: Sequence[Any]) -> list[dict[str, Any]]:
    """Return dataclass records as row dicts (for ``_report_rows`` hooks)."""

    return [
        {spec.name: getattr(record, spec.name) for spec in dataclasses.fields(record)}
        if dataclasses.is_dataclass(record)
        else dict(record)
        for record in records
    ]


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class _TableReport(DebugReport):
    """A tabular ``tl.debug`` result: the report HAS the frame (U42.2).

    ``to_pandas()`` returns a copy of the frame the helper used to return
    (its ``attrs`` included). Never forwards DataFrame methods.
    """

    helper: str
    frame: Any
    subject_text: str
    finding_columns: tuple[str, ...]

    @property
    def report_type(self) -> str:
        """The helper's name (``hot_path``, ``compare``, ...)."""

        return self.helper

    def _report_subject(self) -> str:
        """What this report examined (the card's subject)."""

        return self.subject_text

    def _report_status(self) -> str:
        """The verdict token on the card's lead line."""

        return "empty" if len(self.frame) == 0 else f"{len(self.frame)} rows"

    def _report_findings(self) -> list[str]:
        """One-line findings for the card, most severe first."""

        columns = [column for column in self.finding_columns if column in self.frame.columns]
        findings = []
        for record in self.frame[columns].to_dict(orient="records"):
            findings.append(", ".join(f"{key}={_cell(value)}" for key, value in record.items()))
        return findings

    def to_pandas(self) -> Any:
        """Return a copy of the helper's frame, ``attrs`` included."""

        frame = self.frame.copy()
        frame.attrs = dict(self.frame.attrs)
        return frame

    def to_dict(self) -> dict[str, Any]:
        """Return the lead plus the rows and the frame's ``attrs``."""

        return {
            "report_type": self.report_type,
            "subject": self.subject,
            "status": self.status,
            "rows": [
                {str(key): _plain(value) for key, value in record.items()}
                for record in self.frame.to_dict(orient="records")
            ],
            "attrs": _plain(dict(self.frame.attrs)),
        }

    def __repr__(self) -> str:
        """One-line lead."""

        return f"{self.report_type}({self.subject}) status={self.status}"


def _cell(value: Any) -> str:
    """Render one table cell compactly for a finding line."""

    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def table_report(
    helper: str,
    frame: Any,
    *,
    subject: str,
    finding_columns: Sequence[str],
) -> DebugReport:
    """Wrap a helper's frame in the tabular :class:`DebugReport`.

    Parameters
    ----------
    helper:
        The helper's public name (the report type).
    frame:
        The frame the helper builds.
    subject:
        What the helper examined.
    finding_columns:
        Columns shown on each finding line of the printed card.

    Returns
    -------
    DebugReport
        The tabular report.
    """

    return _TableReport(
        helper=helper,
        frame=frame,
        subject_text=subject,
        finding_columns=tuple(finding_columns),
    )


def trace_subject(trace: Any) -> str:
    """Name a trace for a report subject (its model class)."""

    return str(getattr(trace, "model_class_name", None) or type(trace).__name__)
