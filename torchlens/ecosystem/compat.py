"""The compatibility-window report over the governed ledger (MEMO 3.1/3.2).

``compat_window()`` is the ``tl.compat_window()``-shaped surface: it renders
the ledger rows, the rehydration floor facts, and the promise status as one
frozen report. The report never asserts anything the ledger cannot back --
the promise window NUMBER is FORK F1 (JMT) and renders as pending until
adjudicated, and every named remedy release is covered by the
remedy-actually-loads CI test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._io.compat_ledger import LedgerRow, floor_metadata, ledger_rows

__tl_layer__ = "L8"

#: Normative promise text template (MEMO 3.2); the number is FORK F1.
_PROMISE_TEXT_PENDING = (
    "Every stable TorchLens reader analysis-loads every valid stable artifact "
    "in its governed ledger window for a promised period after that "
    "artifact's exact writer contract retires (window length pending "
    "adjudication; the mechanism is identical under both candidate numbers). "
    "Analysis load preserves and exposes facts present at write time, never "
    "invents later facts, and never imports foreign code. There is NO forward "
    "compatibility promise, ever."
)


@dataclass(frozen=True)
class CompatWindowReport:
    """Frozen compatibility-window report.

    Parameters
    ----------
    runtime:
        Current runtime identity facts (torchlens version, schema ceiling
        and floor, live writer contract digest).
    promise:
        The promise text and window status.
    rows:
        Every governed ledger row plus the generated current-runtime row.
    """

    runtime: dict[str, Any]
    promise: dict[str, Any]
    rows: tuple[LedgerRow, ...]

    def to_markdown(self) -> str:
        """Render the report as a compact markdown table.

        Returns
        -------
        str
            Markdown with the runtime facts, the promise status, and one
            row per governed ledger entry.
        """

        lines = ["# TorchLens artifact compatibility window", ""]
        for key, value in self.runtime.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
        lines.append(f"Promise: {self.promise['text']}")
        lines.append(f"Window: {self.promise['window_status']}")
        lines.append("")
        lines.append("| kind | writer | date | tlspec | support | promised_until | bridge reader |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in self.rows:
            stamp = "pre-tlspec" if row.tlspec_version is None else str(row.tlspec_version)
            lines.append(
                f"| {row.artifact_kind} | {row.writer_release} | {row.writer_date} "
                f"| {stamp} | {row.support_class.value} | {row.promised_until()} "
                f"| {row.bridge_reader or 'none-verified'} |"
            )
        return "\n".join(lines)


def compat_window() -> CompatWindowReport:
    """Build the compatibility-window report from the governed ledger.

    Returns
    -------
    CompatWindowReport
        Runtime identity, promise status, and every governed row. Purely
        derived from ledger data and the live writer contract; reads no
        artifacts and executes nothing.
    """

    from .. import __version__

    floor = floor_metadata()
    rows = ledger_rows()
    runtime = {
        "torchlens_version": __version__,
        "floor_tlspec_version": floor["floor_tlspec_version"],
        "ceiling_tlspec_version": floor["ceiling_tlspec_version"],
        "producer_floor_writer": floor["producer_floor_writer"],
        "writer_contract_digest": rows[-1].writer_contract_digest,
        "golden_corpus_sha256": floor["golden_corpus_sha256"],
    }
    promise = {
        "text": _PROMISE_TEXT_PENDING,
        "window_months": floor["promise_window_months"],
        "window_status": floor["promise_window_status"],
    }
    return CompatWindowReport(runtime=runtime, promise=promise, rows=rows)
