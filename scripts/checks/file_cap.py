"""File-size cap: package files over 800 lines and functions over 60 lines.

Every offender must be listed in ``scripts/checks/file_cap_allowlist.txt``
with its line count today; a listed file or function may shrink but never
grow past its recorded count, and an unlisted offender fails. Entries are
``path<TAB>lines`` for files and ``path::qualname<TAB>lines`` for functions
(``qualname`` joins enclosing class and function names with dots; a function
spans its ``def`` line to its last line, decorators excluded).

Complexity is not measured here: ruff has no function-length rule, and the
fleet complexity gate (cyclomatic and cognitive deltas) covers branching.

``python scripts/checks/file_cap.py --update`` rewrites the allowlist with
today's counts, dropping fixed entries and never raising a recorded count.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import FAIL, OK, Outcome  # noqa: E402

FILE_CAP = 800
FUNCTION_CAP = 60
SCANNED = ("torchlens",)
ALLOWLIST = Path("scripts/checks/file_cap_allowlist.txt")
HEADER = """\
# File-cap allowlist: files over {file_cap} lines and functions over {function_cap} lines,
# each with its line count when it was recorded. This list ONLY SHRINKS: remove an entry
# when it drops under the cap, lower a count when it shrinks, never add or raise one
# without a one-line reason in the commit message. Regenerate with
# `python scripts/checks/file_cap.py --update` (it never raises a recorded count).
"""


def _functions(tree: ast.AST) -> Iterator[tuple[str, int]]:
    stack: list[tuple[ast.AST, str]] = [(tree, "")]
    while stack:
        node, prefix = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                if not isinstance(child, ast.ClassDef):
                    yield name, (child.end_lineno or child.lineno) - child.lineno + 1
                stack.append((child, f"{name}."))
            else:
                stack.append((child, prefix))


def measure(repo: Path) -> dict[str, int]:
    """Return ``{entry: lines}`` for every file and function over its cap."""

    found: dict[str, int] = {}
    for top in SCANNED:
        for path in sorted((repo / top).rglob("*.py")):
            rel = path.relative_to(repo).as_posix()
            text = path.read_text(encoding="utf-8")
            lines = len(text.splitlines())
            if lines > FILE_CAP:
                found[rel] = lines
            for name, length in _functions(ast.parse(text, filename=rel)):
                key = f"{rel}::{name}"
                if length > FUNCTION_CAP and length > found.get(key, 0):
                    found[key] = length
    return found


def read_allowlist(repo: Path) -> dict[str, int]:
    """Parse the allowlist into ``{entry: recorded lines}``."""

    entries: dict[str, int] = {}
    path = repo / ALLOWLIST
    if not path.is_file():
        return entries
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            entry, _, count = line.rpartition("\t")
            entries[entry] = int(count)
    return entries


def check(repo: Path) -> Outcome:
    """Fail on an unlisted offender or a listed one grown past its count."""

    measured, allowed = measure(repo), read_allowlist(repo)
    grown = [
        f"{key}: {lines} lines (allowlist {allowed[key]})"
        if key in allowed
        else f"{key}: {lines} lines (not in allowlist)"
        for key, lines in sorted(measured.items())
        if lines > allowed.get(key, 0)
    ]
    shrinkable = [
        f"{key}: allowlist {limit}, now {measured.get(key, 'under cap')}"
        for key, limit in sorted(allowed.items())
        if measured.get(key, 0) < limit
    ]
    files = sum(1 for key in measured if "::" not in key)
    summary = f"{files} files > {FILE_CAP}, {len(measured) - files} functions > {FUNCTION_CAP}"
    if grown:
        return Outcome(FAIL, f"file-cap: {len(grown)} over cap ({summary})", tuple(grown))
    tail = f"; {len(shrinkable)} allowlist entries can shrink" if shrinkable else ""
    return Outcome(OK, f"file-cap: {summary}, all allowlisted{tail}", tuple(shrinkable))


def update(repo: Path) -> int:
    """Rewrite the allowlist with today's counts, never raising a recorded one."""

    measured, allowed = measure(repo), read_allowlist(repo)
    kept = {
        key: min(lines, allowed.get(key, lines))
        for key, lines in measured.items()
        if not allowed or key in allowed
    }
    body = "".join(f"{key}\t{lines}\n" for key, lines in sorted(kept.items()))
    header = HEADER.format(file_cap=FILE_CAP, function_cap=FUNCTION_CAP)
    (repo / ALLOWLIST).write_text(header + body, encoding="utf-8")
    print(f"wrote {len(kept)} entries to {ALLOWLIST}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--update", action="store_true", help="rewrite the allowlist")
    args = parser.parse_args(argv)
    if args.update:
        return update(args.repo)
    outcome = check(args.repo)
    print(f"[{outcome.status.upper():4}] {outcome.message}")
    for line in outcome.details:
        print(f"       {line}")
    return 1 if outcome.failed else 0


if __name__ == "__main__":
    sys.exit(main())
