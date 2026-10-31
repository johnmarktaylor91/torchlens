"""Environment and model compatibility diagnostics, without executing user models."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

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
        Truth-table rows for the inspected model/input pair.
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

        severity_rank = {"error": 0, "warning": 1, "info": 2, "ok": 3}
        findings = sorted(
            (row for row in self.checks if row.is_detected or row.severity != "ok"),
            key=lambda row: severity_rank.get(row.severity, 3),
        )
        clean = [row for row in self.checks if not (row.is_detected or row.severity != "ok")]
        lines = [
            _cap_line(f"TorchLens doctor report: {self.model_type or 'environment'}"),
            (
                f"PyTorch {self.torch_version}; {len(self.checks)} rows: "
                f"{len(findings)} finding(s), {len(clean)} clean"
            ),
        ]
        for row in findings:
            lines.append(_cap_line(f"- [{row.severity}] {row.label} ({row.status}): {row.details}"))
            if row.suggestion:
                lines.append(_cap_line(f"    fix: {row.suggestion}"))
        if clean:
            lines.append(_cap_line("Clean rows: " + ", ".join(row.label for row in clean)))
        lines.append("Full table: report.show() / report.to_markdown()")
        lines.append(
            "Rows: report.checks / report.check(key); capability dump: report.capability_snapshot()"
        )
        return "\n".join(lines)

    def capability_snapshot(self) -> dict[str, bool]:
        """Return the full runtime capability snapshot (the detail accessor).

        The ``torch_capabilities`` row shows the grouped absences-first
        summary; this returns the complete ``flag -> available`` mapping.

        Returns
        -------
        dict[str, bool]
            Capability flag availability.
        """

        from .._utils import capability_snapshot

        return capability_snapshot()


#: Width cap for one cell in the fixed-width ``show()`` table.
_SHOW_CELL_WIDTH_CAP = 60

#: Width cap for one designed-summary line (bounded-reprs doctrine).
_REPR_LINE_CAP = 160


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


def doctor(model: nn.Module | None = None, inputs: Any = None) -> DoctorReport:
    """Inspect the environment and, optionally, a model and its input tree.

    Parameters
    ----------
    model:
        Optional eager model to inspect; its forward is never executed.
    inputs:
        Optional input tree inspected only when a model is supplied.

    Returns
    -------
    DoctorReport
        Environment checks followed by model compatibility findings, if supplied.
    """

    from .._utils import _environment_checks

    checks = tuple(_environment_row(row) for row in _environment_checks())
    model_type = None
    if model is not None:
        from ..compat._report import _model_checks, _qualified_type_name

        checks += _model_checks(model, inputs)
        model_type = _qualified_type_name(model)
    return DoctorReport(model_type, str(torch.__version__), checks)


_ENVIRONMENT_STATUS: dict[str, tuple[Status, Severity]] = {
    "PASS": ("pass", "ok"),
    "WARN": ("scope", "warning"),
    "FAIL": ("known_broken", "error"),
    "SKIP": ("not_tested", "info"),
}
_ENVIRONMENT_REMEDIES = {
    "pytorch": "Install a supported PyTorch build.",
    "runtime capabilities": "Inspect capability_snapshot() for unavailable runtime integrations.",
    "torch wrapper bindings": "Reinstall wrappers with wrap_torch() and refresh stale references.",
    "mechanical belt": "Run a capture first; avoid pre-wrap references to failed belt probes.",
    "cuda": "Use CPU capture or install a CUDA-enabled PyTorch build on a compatible device.",
    "graphviz": "Install torchlens[viz] and the Graphviz system executable.",
    "safetensors": "Install safetensors to enable portable tensor artifacts.",
    "extras": "Reinstall TorchLens with the optional extras needed for your workflow.",
    "model fingerprint": "Check the fingerprint diagnostic and report the failing environment.",
}


def _environment_row(row: Any) -> DoctorCheck:
    """Adapt an existing environment probe to the unified status vocabulary."""

    status, severity = _ENVIRONMENT_STATUS[row.status]
    key = "torch_capabilities" if row.name == "runtime capabilities" else row.name
    return DoctorCheck(
        key,
        row.name.replace("_", " ").capitalize(),
        status,
        severity,
        status != "pass",
        row.detail,
        _ENVIRONMENT_REMEDIES[row.name] if status != "pass" else "",
    )
