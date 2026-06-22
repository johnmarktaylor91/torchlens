"""Import historical menagerie crawl logs into structured sweep provenance."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from menagerie.ledger import VERIFICATION_DB
from menagerie.provenance import SweepProvenance, record_sweep


MENAGERIE_DIR = Path(__file__).resolve().parents[1]
CRAWL_HISTORY = MENAGERIE_DIR / "data" / "crawl_history.json"
CRAWL_LOG = MENAGERIE_DIR / "CRAWL_LOG.md"
HARVEST_SOURCES = MENAGERIE_DIR / "HARVEST_SOURCES.md"


def import_historical_provenance(
    crawl_history: Path = CRAWL_HISTORY,
    crawl_log: Path = CRAWL_LOG,
    harvest_sources: Path = HARVEST_SOURCES,
    db_path: Path = VERIFICATION_DB,
) -> list[SweepProvenance]:
    """Fold existing historical crawl prose into structured provenance rows.

    Parameters
    ----------
    crawl_history:
        Machine-readable crawl history JSON path.
    crawl_log:
        Markdown crawl log path.
    harvest_sources:
        Markdown source-harvest guide path.
    db_path:
        Verification database path that owns the provenance table.

    Returns
    -------
    list[SweepProvenance]
        Imported provenance rows.
    """

    history = json.loads(crawl_history.read_text(encoding="utf-8"))
    crawl_log_text = crawl_log.read_text(encoding="utf-8")
    harvest_text = harvest_sources.read_text(encoding="utf-8")
    rows = []
    for index, crawl in enumerate(history.get("crawls", []), start=1):
        if not isinstance(crawl, dict):
            continue
        date = str(crawl.get("date", "unknown"))
        name = str(crawl.get("name") or crawl.get("type") or f"crawl-{index}")
        sweep_id = _sweep_id(date, name)
        sources = _sources_from_crawl(crawl)
        notes = "\n\n".join(
            part
            for part in (
                _crawl_notes(crawl),
                _markdown_section_for_date(crawl_log_text, date),
                _harvest_excerpt(harvest_text),
            )
            if part
        )
        rows.append(
            record_sweep(
                sweep_id=sweep_id,
                date=date,
                agent=str(crawl.get("agent", "historical-log-import")),
                sources=sources,
                selection_rule=str(
                    crawl.get(
                        "method",
                        "Best-effort historical import; add notable, genuinely new families.",
                    )
                ),
                families_considered=_result_value(crawl, "families_considered"),
                families_added=_result_value(crawl, "families_added"),
                families_rejected=_result_value(crawl, "families_rejected"),
                git_commit=str(crawl.get("git_commit", "")),
                notes=notes,
                db_path=db_path,
            )
        )
    return rows


def _sweep_id(date: str, name: str) -> str:
    """Build a stable import identifier from date and name.

    Parameters
    ----------
    date:
        Sweep date.
    name:
        Sweep name or type.

    Returns
    -------
    str
        Stable sweep identifier.
    """

    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{date}-{slug or 'crawl'}"


def _sources_from_crawl(crawl: dict[str, Any]) -> list[str]:
    """Extract source or axis labels from one crawl-history entry.

    Parameters
    ----------
    crawl:
        Crawl-history object.

    Returns
    -------
    list[str]
        Source labels for the provenance row.
    """

    sources: list[str] = []
    for key in ("sources", "axes", "languages_covered"):
        value = crawl.get(key)
        if isinstance(value, list):
            sources.extend(str(item) for item in value)
        elif isinstance(value, str):
            sources.append(value)
    if not sources:
        sources.append(str(crawl.get("type", "historical crawl log")))
    return sources


def _crawl_notes(crawl: dict[str, Any]) -> str:
    """Serialize one crawl-history entry as raw preserved notes.

    Parameters
    ----------
    crawl:
        Crawl-history object.

    Returns
    -------
    str
        JSON-formatted raw notes.
    """

    return "Raw crawl_history entry:\n" + json.dumps(crawl, indent=2, sort_keys=True)


def _markdown_section_for_date(text: str, date: str) -> str:
    """Extract the markdown section for a dated crawl log entry.

    Parameters
    ----------
    text:
        Markdown source text.
    date:
        Date heading to extract.

    Returns
    -------
    str
        Matching markdown section or an empty string.
    """

    marker = f"## {date}"
    start = text.find(marker)
    if start < 0:
        return ""
    next_heading = text.find("\n## ", start + len(marker))
    end = len(text) if next_heading < 0 else next_heading
    return "Raw CRAWL_LOG excerpt:\n" + text[start:end].strip()


def _harvest_excerpt(text: str) -> str:
    """Return the harvest-source headings as preserved provenance context.

    Parameters
    ----------
    text:
        Harvest-source markdown.

    Returns
    -------
    str
        Compact raw harvest-source context.
    """

    headings = [line for line in text.splitlines() if line.startswith(("## ", "- **"))]
    return "Raw HARVEST_SOURCES headings:\n" + "\n".join(headings)


def _result_value(crawl: dict[str, Any], key: str) -> int | str | list[str]:
    """Extract a provenance count/list value from historical crawl data.

    Parameters
    ----------
    crawl:
        Crawl-history object.
    key:
        Normalized provenance field name.

    Returns
    -------
    int | str | list[str]
        Best-effort historical value.
    """

    if key in crawl:
        value = crawl[key]
    elif key == "families_added":
        value = crawl.get("result", crawl.get("result_partial", ""))
    elif key == "families_considered":
        value = crawl.get("current_totals", crawl.get("result", crawl.get("result_partial", "")))
    else:
        value = crawl.get("rejected", "")
    if isinstance(value, int | str):
        return value
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return ""


def build_parser() -> argparse.ArgumentParser:
    """Build the provenance importer CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-history", type=Path, default=CRAWL_HISTORY)
    parser.add_argument("--crawl-log", type=Path, default=CRAWL_LOG)
    parser.add_argument("--harvest-sources", type=Path, default=HARVEST_SOURCES)
    parser.add_argument("--db", type=Path, default=VERIFICATION_DB)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the historical provenance importer.

    Parameters
    ----------
    argv:
        Optional argument vector.

    Returns
    -------
    int
        Process exit status.
    """

    args = build_parser().parse_args(argv)
    rows = import_historical_provenance(
        crawl_history=args.crawl_history,
        crawl_log=args.crawl_log,
        harvest_sources=args.harvest_sources,
        db_path=args.db,
    )
    print(f"imported_sweep_provenance_rows={len(rows)}")
    for row in rows:
        print(f"{row.date}\t{row.sweep_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
