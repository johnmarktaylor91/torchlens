"""The common report protocol (F10; lovely matrix "report objects" row).

One lead for every report-shaped object: type, subject, status, severity
counts, coverage, truncation -- then at most five findings and an EXACT
remainder. Helpers RETURN report objects that HAVE ``.to_pandas()``; they
are never bare DataFrames (a DataFrame elides its best column and carries
no honesty facts). ``tl.debug``'s three dialects migrate onto this
protocol under their owning lanes (lovely section 14 ownership seam); the
protocol and its conformance test land here so the target shape is law
before the migrations.

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

#: Findings shown inline on a report card; the exact remainder always prints.
REPORT_MAX_FINDINGS = 5


@runtime_checkable
class BoundedReport(Protocol):
    """Structural protocol for report-shaped objects (the common lead).

    Conforming objects expose the lead facts as attributes and render a
    bounded card via ``__str__`` whose first line is their one-line repr;
    ``to_pandas()`` is the tabular exit (the object HAS a table, it never
    IS one).
    """

    report_type: str
    subject: str
    status: str

    def to_pandas(self) -> Any:
        """Return the tabular projection of this report."""
        ...


@dataclass(frozen=True)
class ReportLead:
    """The common lead block every report card opens with.

    Parameters mirror the lovely matrix row: type/subject/status/severity
    counts/coverage/truncation. ``severity_counts`` keys are free-form
    severity tokens; ``coverage`` states what was and was not examined
    (evidence, never vibes); ``truncation`` is the exact-remainder note
    when the card bounded its findings.
    """

    report_type: str
    subject: str
    status: str
    severity_counts: dict[str, int] = field(default_factory=dict)
    coverage: str | None = None
    truncation: str | None = None

    def lead_line(self) -> str:
        """Render the one-line lead (a report card's line 1)."""

        parts = [f"{self.report_type}({self.subject})", f"status={self.status}"]
        if self.severity_counts:
            counts = ", ".join(
                f"{token}={count}" for token, count in sorted(self.severity_counts.items())
            )
            parts.append(f"[{counts}]")
        return " ".join(parts)


def render_report_card(
    lead: ReportLead,
    findings: list[str],
    *,
    exits: tuple[str, ...] = (".to_pandas()",),
) -> str:
    """Assemble a bounded report card: lead + <=5 findings + exact remainder.

    Parameters
    ----------
    lead:
        The common lead block.
    findings:
        Finding one-liners in severity order (kept from the front).
    exits:
        Accessor exits named on the ``More:`` line.

    Returns
    -------
    str
        The bounded card; the remainder is exact, never a bare ellipsis.
    """

    lines = [lead.lead_line()]
    if lead.coverage:
        lines.append(f"  coverage: {lead.coverage}")
    shown = findings[:REPORT_MAX_FINDINGS]
    lines.extend(f"  - {finding}" for finding in shown)
    remainder = len(findings) - len(shown)
    if remainder > 0:
        lines.append(f"  ... {remainder} more findings (see {exits[0]})")
    if lead.truncation:
        lines.append(f"  truncation: {lead.truncation}")
    lines.append(f"  More: {'  '.join(exits)}")
    return "\n".join(lines)
