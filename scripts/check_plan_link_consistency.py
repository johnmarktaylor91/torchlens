"""Check structural references inside the canonical documentation plan."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SECTION_2_6_PATTERN = re.compile(r"§2\.6|Section 2\.6")
PLAN_LINE_RANGE_PATTERN = re.compile(r"plan lines (\d+)-(\d+)")
COMPARISON_TABLE_PHRASE_PATTERN = re.compile(r"comparison table", re.IGNORECASE)
FIRST_COMPARISON_ROW = "| Captures every eager op"
LAST_COMPARISON_ROW = "| Mature transformer-internals shortcuts"


@dataclass(frozen=True)
class CitationError:
    """Invalid plan citation discovered by the checker.

    Attributes
    ----------
    line_number:
        One-indexed plan line where the citation paragraph starts.
    message:
        Human-readable failure detail.
    """

    line_number: int
    message: str


def _paragraphs(lines: list[str]) -> list[tuple[int, str]]:
    """Split Markdown into blank-line-delimited paragraphs.

    Parameters
    ----------
    lines:
        Markdown file lines without trailing newline normalization.

    Returns
    -------
    list[tuple[int, str]]
        ``(start_line_number, paragraph_text)`` entries.
    """

    paragraphs: list[tuple[int, str]] = []
    start_line: int | None = None
    parts: list[str] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            if start_line is not None:
                paragraphs.append((start_line, " ".join(parts)))
            start_line = None
            parts = []
            continue
        if start_line is None:
            start_line = index
        parts.append(line.strip())
    if start_line is not None:
        paragraphs.append((start_line, " ".join(parts)))
    return paragraphs


def _history_line_numbers(lines: list[str]) -> set[int]:
    """Return line numbers that belong to historical §12 audit prose.

    Parameters
    ----------
    lines:
        Markdown file lines.

    Returns
    -------
    set[int]
        One-indexed line numbers at or after the first ``### §12.3``
        round-integration-log heading.
    """

    history_lines: set[int] = set()
    in_history = False
    for index, line in enumerate(lines, start=1):
        if line.startswith("## Appendix A"):
            in_history = False
        elif re.match(r"^### §12\.(?:[3-9]|\d{2,})\s+Round", line):
            in_history = True
        if in_history:
            history_lines.add(index)
    return history_lines


def _range_contains_row(lines: list[str], start_line: int, end_line: int, prefix: str) -> bool:
    """Return whether a line range contains a Markdown table row prefix.

    Parameters
    ----------
    lines:
        Markdown file lines.
    start_line:
        One-indexed inclusive range start.
    end_line:
        One-indexed inclusive range end.
    prefix:
        Required row prefix.

    Returns
    -------
    bool
        Whether the row prefix is present inside the cited range.
    """

    if start_line < 1 or end_line < start_line:
        return False
    selected = lines[start_line - 1 : end_line]
    return any(line.startswith(prefix) for line in selected)


def validate_s2_6_citation_range(lines: list[str], start_line: int, end_line: int) -> list[str]:
    """Validate a §2.6 comparison-table line citation.

    Parameters
    ----------
    lines:
        Markdown file lines.
    start_line:
        One-indexed inclusive range start from the citation.
    end_line:
        One-indexed inclusive range end from the citation.

    Returns
    -------
    list[str]
        Failure messages. Empty means the range contains both required
        §2.6 table boundary data rows.
    """

    errors: list[str] = []
    if not _range_contains_row(lines, start_line, end_line, FIRST_COMPARISON_ROW):
        errors.append(f"range {start_line}-{end_line} misses first §2.6 data row")
    if not _range_contains_row(lines, start_line, end_line, LAST_COMPARISON_ROW):
        errors.append(f"range {start_line}-{end_line} misses last §2.6 data row")
    return errors


def find_invalid_s2_6_citations(lines: list[str]) -> list[CitationError]:
    """Find stale §2.6 comparison-table line-range citations.

    Parameters
    ----------
    lines:
        Markdown file lines.

    Returns
    -------
    list[CitationError]
        Invalid active citations. Historical §12 audit prose is ignored.
    """

    history_lines = _history_line_numbers(lines)
    errors: list[CitationError] = []
    for start_line, paragraph in _paragraphs(lines):
        if start_line in history_lines:
            continue
        if not SECTION_2_6_PATTERN.search(paragraph):
            continue
        if not COMPARISON_TABLE_PHRASE_PATTERN.search(paragraph):
            continue
        for match in PLAN_LINE_RANGE_PATTERN.finditer(paragraph):
            line_start = int(match.group(1))
            line_end = int(match.group(2))
            range_errors = validate_s2_6_citation_range(lines, line_start, line_end)
            for message in range_errors:
                errors.append(CitationError(line_number=start_line, message=message))
    return errors


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv:
        Command-line argument list excluding the program name.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plan_path",
        nargs="?",
        default=".research/docs-plan-megasprint_PLAN.md",
        type=Path,
        help="Path to the canonical plan Markdown file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run plan link-consistency checks.

    Parameters
    ----------
    argv:
        Optional argument list excluding the program name.

    Returns
    -------
    int
        Process exit code.
    """

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    lines = args.plan_path.read_text().splitlines()
    errors = find_invalid_s2_6_citations(lines)
    if not errors:
        return 0
    for error in errors:
        print(f"{args.plan_path}:{error.line_number}: {error.message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
