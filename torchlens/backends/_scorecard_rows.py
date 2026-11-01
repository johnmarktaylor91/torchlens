"""``backend_scorecard:<name>`` rows for ``tl.compat.report()``.

One row per registered backend, built from the scorecard record shipped at
``torchlens/backends/<name>/scorecard.json`` and the backend's registered
``FrameworkRequirement``. The rows read the shipped JSON only: no framework
is imported and nothing executes, so the compat report stays cheap and
side-effect free. The same builder serves the ``tl.report`` merge.

The shipped record is a measured fact (regenerated and checked by the
backend's pinned CI leg), never hand-edited; a backend with no record reads
``not_tested`` here.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..compat._report import CompatRow, Severity, Status
    from ..conformance._scorecard import PreviewScorecard

__tl_layer__ = "L2"

#: File name of the shipped record inside each backend package.
SCORECARD_FILE_NAME = "scorecard.json"

#: Row key prefix; the full key is ``backend_scorecard:<name>``.
ROW_KEY_PREFIX = "backend_scorecard:"

#: The compat row statuses this builder writes, in precedence order.
ROW_STATUSES: tuple[str, ...] = ("not_tested", "known_broken", "scope", "pass")


def shipped_scorecard_path(backend: str) -> Path:
    """Path of the shipped record for ``backend`` (it may not exist)."""

    return Path(__file__).resolve().parent / backend / SCORECARD_FILE_NAME


def read_shipped_scorecard(backend: str) -> dict[str, Any] | None:
    """The shipped record for ``backend`` as a dict, or ``None`` when absent.

    A record whose body is not a JSON object is treated as absent; a
    malformed file is a packaging defect the scorecard ``--check`` leg
    catches, not something the compat report should raise on.
    """

    from .._io import _json

    path = shipped_scorecard_path(backend)
    if not path.is_file():
        return None
    try:
        record = _json.read_bounded(path)
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def scorecard_from_record(record: dict[str, Any]) -> PreviewScorecard:
    """Rebuild a ``PreviewScorecard`` from a shipped record.

    Only fields the dataclass declares are passed, so a record written by a
    later scorecard version (extra keys) still loads; cells become
    ``ScorecardCell`` rows.
    """

    from ..conformance._scorecard import PreviewScorecard, ScorecardCell

    known = {field.name for field in dataclass_fields(PreviewScorecard)}
    payload = {key: value for key, value in record.items() if key in known}
    cell_fields = {field.name for field in dataclass_fields(ScorecardCell)}
    payload["cells"] = tuple(
        ScorecardCell(**{key: value for key, value in cell.items() if key in cell_fields})
        for cell in record.get("cells", ())
        if isinstance(cell, dict)
    )
    payload.setdefault("counts", {})
    payload.setdefault("highest_tier", "")
    for name in ("scorecard_version", "backend", "framework_version", "suite_version"):
        payload.setdefault(name, "")
    payload.setdefault("torchlens_version", "")
    return PreviewScorecard(**payload)


def backend_scorecards() -> tuple[PreviewScorecard, ...]:
    """The shipped records of every registered backend that has one."""

    from .registry import registered_backend_specs

    cards = []
    for spec in registered_backend_specs():
        record = read_shipped_scorecard(spec.name)
        if record is not None:
            cards.append(scorecard_from_record(record))
    return tuple(cards)


def backend_scorecard_rows() -> tuple[CompatRow, ...]:
    """One ``backend_scorecard:<name>`` row per registered backend."""

    from .registry import registered_backend_specs

    return tuple(_row_for_spec(spec) for spec in registered_backend_specs())


def _row_for_spec(spec: Any) -> CompatRow:
    """Build the row for one registered spec from its shipped record."""

    from ..compat._report import CompatRow

    record = read_shipped_scorecard(spec.name)
    framework = _framework_requirement(spec)
    installed = framework.installed_version() if framework is not None else None
    maturity = str(_record_value(record, "maturity") or getattr(spec, "maturity", "preview"))
    counts = _counts(record)
    status = _status_for(record, counts, installed=installed)
    extra = getattr(framework, "extra", "") if framework is not None else ""
    return CompatRow(
        key=f"{ROW_KEY_PREFIX}{spec.name}",
        label=f"{spec.name} backend scorecard",
        status=status,
        severity=_severity_for(status, maturity),
        is_detected=installed is not None,
        details=_details(record, counts, maturity, framework, installed),
        suggestion=_suggestion(record, status, installed, spec.name, extra),
    )


def _framework_requirement(spec: Any) -> Any | None:
    """The spec's ``FrameworkRequirement`` through its protocol shell, if any.

    The shells are import-light (a protocol object never imports its
    framework), so this stays side-effect free.
    """

    factory = getattr(spec, "protocol", None)
    if factory is None:
        return None
    try:
        return getattr(factory(), "framework", None)
    except Exception:  # noqa: BLE001 - a broken factory is reported as "unknown", never raised
        return None


def _record_value(record: dict[str, Any] | None, key: str) -> Any:
    """``record[key]`` or ``None``."""

    return None if record is None else record.get(key)


def _counts(record: dict[str, Any] | None) -> dict[str, int]:
    """Integer cell counts from the record, every state present."""

    raw = _record_value(record, "counts")
    counts = dict.fromkeys(("passed", "failed", "unsupported", "not_run", "denominator"), 0)
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, int) and not isinstance(value, bool):
                counts[str(key)] = value
    return counts


def _status_for(
    record: dict[str, Any] | None, counts: dict[str, int], *, installed: str | None
) -> Status:
    """The row status, in this order of precedence.

    No shipped record reads ``not_tested``; any failed cell reads
    ``known_broken``; a framework that is not installed here reads ``scope``;
    a clean record at C2 reads ``pass``; a clean record below C2 reads
    ``not_tested`` when cells did not run, else ``scope`` (typed refusals
    bound the tier).
    """

    if record is None:
        return "not_tested"
    if counts["failed"] > 0:
        return "known_broken"
    if installed is None:
        return "scope"
    if _record_value(record, "highest_tier") == "C2":
        return "pass"
    return "not_tested" if counts["not_run"] > 0 else "scope"


def _severity_for(status: str, maturity: str) -> Severity:
    """Row severity: a stable backend's failures are errors, a preview's warnings."""

    if status == "pass":
        return "ok"
    if status == "known_broken":
        return "error" if maturity == "stable" else "warning"
    return "info"


def _details(
    record: dict[str, Any] | None,
    counts: dict[str, int],
    maturity: str,
    framework: Any | None,
    installed: str | None,
) -> str:
    """The details cell in the SPEC 4.6 shape."""

    if record is None:
        cells = "no shipped scorecard"
    else:
        cells = (
            f"{counts['passed']}/{counts['denominator']} cells passed, "
            f"{counts['unsupported']} unsupported, {counts['failed']} failed"
        )
    pinned = getattr(framework, "pinned", None) if framework is not None else None
    minimum = getattr(framework, "minimum", "?") if framework is not None else "?"
    below = getattr(framework, "below", None) if framework is not None else None
    tested_through = getattr(framework, "tested_through", "") if framework is not None else ""
    upper = (
        below
        if below is not None
        else (f"tested through {tested_through}" if tested_through else "none")
    )
    return (
        f"{maturity}; {cells}; pinned {pinned if pinned is not None else 'unpinned'}, "
        f"installed {installed if installed is not None else 'absent'}, "
        f"supported [{minimum}, {upper})"
    )


def _suggestion(
    record: dict[str, Any] | None,
    status: str,
    installed: str | None,
    backend: str,
    extra: str,
) -> str:
    """Install extra when the framework is absent; else the first failed cell."""

    if installed is None:
        target = f"torchlens[{extra}]" if extra else backend
        return f'install the framework: pip install "{target}"'
    if status == "known_broken" and record is not None:
        for cell in record.get("cells", ()):
            if isinstance(cell, dict) and cell.get("state") == "failed":
                family = cell.get("family") or ""
                where = f" ({family})" if family else ""
                detail = str(cell.get("detail") or "")[:120]
                return f"first failed cell: {cell.get('case_id')}{where}: {detail}".rstrip(": ")
    if record is None:
        return f"run python -m torchlens.conformance.scorecard --backend {backend} --write on the pinned leg"
    return ""
