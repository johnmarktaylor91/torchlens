"""Doctor report types: the compatibility truth-table row and the report that holds them.

A basis module (layer 0): the rows are built by ``torchlens.report.doctor`` and by the
backend scorecard, and read by ``torchlens.compat``. Keeping the types here lets the
scorecard and the compatibility report import them eagerly without reaching the report
package. The public path stays ``torchlens.report.DoctorCheck`` / ``DoctorReport``.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

__tl_layer__ = "L0"

#: Full capability-snapshot provider, registered by ``torchlens.report._doctor`` at import
#: (every ``DoctorReport`` is built there); until then the torch-only snapshot answers.
_CAPABILITY_SNAPSHOT_PROVIDER: Callable[[], dict[str, bool]] | None = None


def register_capability_snapshot_provider(provider: Callable[[], dict[str, bool]]) -> None:
    """Install the function ``DoctorReport.capability_snapshot`` delegates to."""

    global _CAPABILITY_SNAPSHOT_PROVIDER
    _CAPABILITY_SNAPSHOT_PROVIDER = provider


def _capability_snapshot() -> dict[str, bool]:
    """Return the registered snapshot; empty before ``torchlens.report`` registers its provider.

    This module is layer 0 and imports nothing from the package, so the probe
    lives in the report layer and reaches here through the registration slot.
    """

    provider = _CAPABILITY_SNAPSHOT_PROVIDER
    if provider is None:
        return {}
    return provider()


Status = Literal["pass", "known_broken", "scope", "not_tested"]
Severity = Literal["ok", "info", "warning", "error"]


@dataclass(frozen=True)
class DoctorCheck:
    """One row in the TorchLens compatibility truth table.

    Attributes
    ----------
    key:
        Stable machine-readable row key.
    label:
        Human-readable row label.
    status:
        Compatibility status for this model/input pair.
    severity:
        User-facing severity for the row.
    is_detected:
        Whether the row's condition was detected in the supplied model/input.
    details:
        Explanation of the finding.
    suggestion:
        Suggested workaround or next action.

    Examples
    --------
    >>> report = tl.report.doctor()
    >>> row = report.checks[0]
    >>> isinstance(row, DoctorCheck), row.status in ("pass", "known_broken", "scope", "not_tested")
    (True, True)
    """

    key: str
    label: str
    status: Status
    severity: Severity
    is_detected: bool
    details: str
    suggestion: str = ""

    def __repr__(self) -> str:
        """Return a bounded row summary; full diagnostics stay on attributes."""

        return _cap_line(f"DoctorCheck({self.key!r}, {self.status}, {self.severity})")


@dataclass(frozen=True)
class DoctorReport:
    """Compatibility report returned by :func:`torchlens.report.doctor`.

    Attributes
    ----------
    model_type:
        Qualified type name for the inspected model.
    torch_version:
        PyTorch version used for the inspection.
    checks:
        Truth-table rows: environment checks, one ``backend_scorecard:<name>``
        row per registered backend (its shipped conformance record), then the
        model/input findings when a model was supplied.

    Examples
    --------
    >>> report = tl.report.doctor(model, x)
    >>> isinstance(report, DoctorReport)
    True
    >>> report.model_type
    'torch.nn.modules.container.Sequential'
    >>> len(report.checks) > 0
    True
    """

    model_type: str | None
    torch_version: str
    checks: tuple[DoctorCheck, ...]

    def check(self, key: str) -> DoctorCheck:
        """Return one report row by stable key.

        Parameters
        ----------
        key:
            Row key to look up.

        Returns
        -------
        DoctorCheck
            Matching report row.

        Raises
        ------
        KeyError
            If ``key`` is not present in this report.
        """

        for report_row in self.checks:
            if report_row.key == key:
                return report_row
        raise KeyError(key)

    def to_markdown(self) -> str:
        """Render this report as a GitHub-flavored Markdown table.

        Returns
        -------
        str
            Markdown representation of the report.
        """

        lines = [
            f"### TorchLens doctor report for `{self.model_type or 'environment'}`",
            "",
            f"- PyTorch: `{self.torch_version}`",
            f"- Rows: {len(self.checks)}",
            "",
            "| Row | Status | Severity | Detected | Details | Suggestion |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in self.checks:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_markdown(row.label),
                        f"`{row.status}`",
                        f"`{row.severity}`",
                        "yes" if row.is_detected else "no",
                        _escape_markdown(row.details),
                        _escape_markdown(row.suggestion),
                    ]
                )
                + " |"
            )
        return "\n".join(lines)

    def _render_table(self) -> str:
        """Render this report as a fixed-width text table.

        Long ``Details``/``Suggestion`` cells wrap inside a capped column
        width instead of inflating every table line (a ~1,900-character data
        cell used to yield 2,073-character lines on every renderer).

        Returns
        -------
        str
            Text table suitable for terminals and notebook display.
        """

        headers = ("Row", "Status", "Severity", "Detected", "Details", "Suggestion")
        body = [
            (
                row.label,
                row.status,
                row.severity,
                "yes" if row.is_detected else "no",
                row.details,
                row.suggestion,
            )
            for row in self.checks
        ]
        widths = tuple(
            min(width, _SHOW_CELL_WIDTH_CAP) for width in _column_widths([headers, *body])
        )
        lines = [
            f"TorchLens doctor report: {self.model_type or 'environment'}",
            f"PyTorch: {self.torch_version}",
            "",
            _format_table_line(headers, widths),
            _format_table_line(tuple("-" * width for width in widths), widths),
        ]
        for row_values in body:
            lines.extend(_format_wrapped_table_lines(row_values, widths))
        return "\n".join(lines)

    def show(self) -> str:
        """Print the fixed-width table and return the same text."""

        text = self._render_table()
        print(text)
        return text

    def __str__(self) -> str:
        """Return the fixed-width table as text."""

        return self._render_table()

    def __repr__(self) -> str:
        """Return a bounded findings-first summary."""

        return _cap_line(
            f"DoctorReport {self.model_type or 'environment'}: {len(self.checks)} rows"
        )

    def capability_snapshot(self) -> dict[str, bool]:
        """Return the full runtime capability snapshot (the detail accessor).

        The ``torch_capabilities`` row shows the grouped absences-first
        summary; this returns the complete ``flag -> available`` mapping.

        Returns
        -------
        dict[str, bool]
            Capability flag availability.
        """

        return _capability_snapshot()


#: Width cap for one cell in the fixed-width ``show()`` table.
_SHOW_CELL_WIDTH_CAP = 60

#: Width cap for one designed-summary line (bounded-reprs doctrine).
_REPR_LINE_CAP = 100


def _cap_line(line: str) -> str:
    """Return ``line`` truncated to the repr cap with a disclosed ellipsis."""

    if len(line) <= _REPR_LINE_CAP:
        return line
    return line[: _REPR_LINE_CAP - 3] + "..."


def _format_wrapped_table_lines(row: Sequence[str], widths: Sequence[int]) -> list[str]:
    """Format one logical table row as one or more width-capped lines.

    Parameters
    ----------
    row:
        Cell values.
    widths:
        Capped column widths.

    Returns
    -------
    list[str]
        Physical lines; cells longer than their column wrap onto
        continuation lines in their own column.
    """

    wrapped_cells = [
        textwrap.wrap(value, width=widths[index]) or [""] for index, value in enumerate(row)
    ]
    line_count = max(len(cell) for cell in wrapped_cells)
    lines = []
    for line_index in range(line_count):
        parts = tuple(cell[line_index] if line_index < len(cell) else "" for cell in wrapped_cells)
        lines.append(_format_table_line(parts, widths).rstrip())
    return lines


def _escape_markdown(value: str) -> str:
    """Escape table separators in Markdown cells.

    Parameters
    ----------
    value:
        Cell value.

    Returns
    -------
    str
        Escaped cell value.
    """

    return value.replace("|", "\\|").replace("\n", " ")


def _column_widths(rows: Sequence[Sequence[str]]) -> tuple[int, ...]:
    """Compute fixed-width table column sizes.

    Parameters
    ----------
    rows:
        Table rows.

    Returns
    -------
    tuple[int, ...]
        Width for each column.
    """

    column_count = len(rows[0])
    return tuple(max(len(row[index]) for row in rows) for index in range(column_count))


def _format_table_line(row: Sequence[str], widths: Sequence[int]) -> str:
    """Format one fixed-width table row.

    Parameters
    ----------
    row:
        Cell values.
    widths:
        Column widths.

    Returns
    -------
    str
        Formatted row.
    """

    return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
