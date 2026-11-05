"""Write glossary keyword rows for public parameters the glossary does not name yet.

Reads the rows ``glossary_params.py --dump`` writes and renders each missing
``(callable, parameter)`` pair in the glossary's keyword-row format::

    | `k=` | keyword of `Bundle.rank_changes` (keyword-only; default `10`): Maximum rows ... |

A keyword that already has a row gains clauses at the end of that row; a keyword with no row
gets a new row in the last ``### Keywords`` table, which stays sorted by spelling. Rows
describe the code as it is (kind, default and the docstring's meaning); they rename nothing.

Usage: ``PYTHONPATH=tools python -m name_lockstep.glossary_backfill DUMP.json [--glossary P]``.

``--prune-records`` (no dump needed) drops every record class (``records.py``) from the owner
runs of the keyword rows: records are built by TorchLens, so no keyword row names their
constructors; a ``field and constructor keyword of`` clause keeps its owners as ``field of``.
``--prune-stale`` drops, from the rows naming them, the keywords the dumped code no longer
takes (a removed or renamed parameter) before the backfill adds the new spellings.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from name_lockstep.glossary_params import _KEYWORD_ROW, _OWNER_RUN, _SPAN, _names, missing, stale
from name_lockstep.records import is_record, record_classes

_RST_LITERAL = re.compile(r"``([^`]+)``")
_SPHINX_ROLE = re.compile(
    r":(?:py:)?(?:class|func|meth|attr|data|exc|obj):`~?([^`<]*?)\s*(?:<([^`>]+)>)?`"
)
_INHERITED = " and the classes that inherit that constructor"
_PARENTHETICAL = re.compile(r"\s*\(([^()]*)\)")
_ID_LINT = Path(__file__).resolve().parents[2] / "scripts" / "checks" / "id_lint.py"
#: A clause boundary inside a keyword row's meaning cell.
_CLAUSE_SPLIT = re.compile(r";\s+(?=(?:field and )?(?:constructor )?keyword of `)")
_CLAUSE_HEAD = re.compile(r"^(field and )?(constructor )?keyword of ")


@functools.cache
def _internal_id() -> re.Pattern[str]:
    """Return one pattern for every id family ``scripts/checks/id_lint.py`` counts."""

    spec = importlib.util.spec_from_file_location("_glossary_id_lint", _ID_LINT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_ID_LINT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return re.compile("|".join(f"(?:{pattern})" for pattern in module.FAMILIES.values()))


def _without_ids_in_parentheses(match: re.Match[str]) -> str:
    """Drop the ``;`` parts of a parenthetical that cite an internal id."""

    kept = [part.strip() for part in match.group(1).split(";")]
    kept = [part for part in kept if part and not _internal_id().search(part)]
    return f" ({'; '.join(kept)})" if kept else ""


def strip_internal_ids(text: str) -> str:
    """Return docstring prose without internal planning ids (memo items, findings, lanes).

    The public glossary must not cite private planning ids; ``id-lint`` ratchets them down.
    A parenthetical loses its id-citing parts; a bare id elsewhere is removed.
    """

    if not _internal_id().search(text):
        return text
    text = _PARENTHETICAL.sub(_without_ids_in_parentheses, text)
    text = _internal_id().sub("\0", text)  # mark each removed id to tidy only around it
    text = re.sub(r"\s*\0[\0/,\s]*(?=[,;)]|\.(?!\.))", "", text)
    return re.sub(r"\s*\0[\0/,\s]*", " ", text).strip()


def _role_name(match: re.Match[str]) -> str:
    """Render a Sphinx cross-reference by its last name, as Sphinx shows ``~`` references.

    Docstring targets are often private or moved module paths; the glossary names the object,
    and only public ``tl.*`` spellings it can resolve.
    """

    title, target = match.group(1), match.group(2)
    return f"`{title if target else title.rsplit('.', 1)[-1]}`"


def _clean(meaning: str) -> str:
    """Return a docstring meaning as one markdown table cell, without internal ids."""

    text = _SPHINX_ROLE.sub(_role_name, " ".join(meaning.split()))
    text = strip_internal_ids(_RST_LITERAL.sub(r"`\1`", text))
    return text.replace("|", "\\|").strip().removesuffix(".")


def _details(row: dict[str, Any]) -> str:
    """Return the ``(keyword-only; default `x`)`` note for a row."""

    parts = ["keyword-only"] if row["kind"] == "keyword_only" else []
    default = row["default"]
    if default is None:
        parts.append("required")
    elif default == "MISSING" or default.startswith("<"):
        parts.append("optional")  # an unset sentinel: omitting it is not a value
    else:
        parts.append(f"default `{default}`")
    return f" ({'; '.join(parts)})"


def _owner_rows(param_gaps: list[dict[str, Any]], by_pair: dict[tuple[str, str], dict]):
    """Collapse gaps onto the callable that defines each signature.

    Parameters
    ----------
    param_gaps
        Missing rows for one keyword.
    by_pair
        Every dumped row keyed on ``(callable, param)``.

    Returns
    -------
    list[tuple[dict[str, Any], bool]]
        ``(defining row, inherited)``; ``inherited`` is True when subclasses inherit it.
    """

    owners: dict[str, tuple[dict[str, Any], bool]] = {}
    for gap in param_gaps:
        owner = gap["defined_by"]
        source = by_pair.get((owner, gap["param"]), gap)
        inherited = owners.get(owner, (source, False))[1] or gap["callable"] != owner
        owners[owner] = (source, inherited)
    return list(owners.values())


def render_clauses(param_gaps: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    """Return the ``keyword of ...`` clauses naming every gap of one keyword.

    Parameters
    ----------
    param_gaps
        Missing rows for one keyword.
    rows
        Every dumped row (meanings and defaults of defining callables).

    Returns
    -------
    str
        Clauses joined by ``; `` with owners that share kind, default and meaning grouped.
    """

    by_pair = {(r["callable"], r["param"]): r for r in rows}
    groups: dict[tuple[str, str, str, bool], list[str]] = {}
    for row, inherited in _owner_rows(param_gaps, by_pair):
        word = "constructor keyword of" if row["constructor"] else "keyword of"
        key = (word, _details(row), _clean(row["meaning"]), inherited)
        groups.setdefault(key, []).append(row["callable"])
    clauses = []
    for (word, details, meaning, inherited), owners in groups.items():
        names = ", ".join(f"`{owner}`" for owner in sorted(owners))
        tail = _INHERITED if inherited else ""
        clauses.append(f"{word} {names}{tail}{details}{': ' + meaning if meaning else ''}")
    return "; ".join(sorted(clauses, key=lambda c: c.split("`")[1]))


def _last_keyword_table(lines: list[str]) -> tuple[int, int]:
    """Return the ``[start, end)`` line span of the rows of the last ``### Keywords`` table."""

    heading = max(i for i, line in enumerate(lines) if line.strip() == "### Keywords")
    header = next(i for i in range(heading, len(lines)) if lines[i].startswith("|"))
    start = header + 2  # header row, separator row
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return start, end


def backfill(glossary: str, rows: list[dict[str, Any]]) -> tuple[str, int, int]:
    """Return the glossary with every missing pair named, plus extended and new row counts.

    Parameters
    ----------
    glossary
        Glossary text.
    rows
        Every dumped row.

    Returns
    -------
    tuple[str, int, int]
        New text, number of existing rows extended, number of rows added.
    """

    gaps: dict[str, list[dict[str, Any]]] = {}
    for gap in missing(rows, glossary):
        gaps.setdefault(gap["param"], []).append(gap)
    lines = glossary.splitlines()
    start, end = _last_keyword_table(lines)
    extended = 0
    for index, line in _rows_to_extend(lines, start, end, gaps):
        lines[index] = f"{line.rstrip().removesuffix('|').rstrip()}; " + (
            f"{render_clauses(gaps.pop(_name(line)), rows)} |"
        )
        extended += 1
    added = [f"| `{name}=` | {render_clauses(found, rows)} |" for name, found in gaps.items()]
    table = sorted(lines[start:end] + added, key=_sort_key)
    lines[start:end] = table
    return "\n".join(lines) + "\n", extended, len(added)


def _prune_clause(clause: str, gone: Callable[[str], bool]) -> list[str]:
    """Return the clauses that replace one clause once ``gone`` owners leave its run."""

    head = _CLAUSE_HEAD.match(clause)
    run = _OWNER_RUN.match(clause[head.end() - len("keyword of ") :]) if head else None
    if head is None or run is None:
        return [clause]
    owners = [span.strip() for span in _SPAN.findall(run.group(1))]
    kept = [o for o in owners if not gone(o.removesuffix("()"))]
    if len(kept) == len(owners):
        return [clause]
    tail = clause[head.end() - len("keyword of ") + run.end() :].strip()
    out: list[str] = []
    if head.group(1):
        dropped = [o for o in owners if o not in kept]
        out.append("field of " + ", ".join(f"`{o}`" for o in dropped))
    if kept:
        out.append(f"{head.group(0)}{', '.join(f'`{o}`' for o in kept)} {tail}".rstrip())
    return out


def _prune(glossary: str, gone: Callable[[str, str], bool]) -> tuple[str, int, int]:
    """Return the glossary with every owner ``gone(keyword, owner)`` out of the keyword rows.

    A ``field and constructor keyword of`` clause keeps its dropped owners as ``field of``; a
    row whose every clause named only dropped owners is removed.
    """

    lines: list[str] = []
    edited = removed = 0
    for line in glossary.splitlines():
        match = _KEYWORD_ROW.match(line.strip())
        if not match:
            lines.append(line)
            continue
        name = match.group(1)
        clauses = _CLAUSE_SPLIT.split(match.group(2).strip())
        kept = [
            c
            for clause in clauses
            for c in _prune_clause(clause, lambda o, name=name: gone(name, o))
        ]
        if kept == clauses:
            lines.append(line)
        elif kept:
            lines.append(f"| `{name}=` | {'; '.join(kept)} |")
            edited += 1
        else:
            removed += 1
    return "\n".join(lines) + "\n", edited, removed


def prune_records(glossary: str) -> tuple[str, int, int]:
    """Return the glossary with no keyword row naming a record class, plus edit counts.

    Parameters
    ----------
    glossary
        Glossary text; its records section says which classes are records.

    Returns
    -------
    tuple[str, int, int]
        New text, rows edited, rows removed (every clause named only records).
    """

    records = record_classes(glossary)
    return _prune(glossary, lambda _name, owner: is_record(owner, records))


def prune_stale(glossary: str, rows: list[dict[str, Any]]) -> tuple[str, int, int]:
    """Return the glossary with no keyword row naming a keyword its callable lacks.

    Parameters
    ----------
    glossary
        Glossary text.
    rows
        Every dumped row (the code's public parameters).

    Returns
    -------
    tuple[str, int, int]
        New text, rows edited, rows removed.
    """

    aliases = {row["callable"]: row["aliases"] for row in rows}
    gone: dict[str, list[list[str]]] = {}
    for callable_, name in stale(rows, glossary):
        gone.setdefault(name, []).append(aliases[callable_])
    return _prune(glossary, lambda name, owner: any(_names(owner, a) for a in gone.get(name, [])))


_ENTRY_ROW = re.compile(r"^\|\s*`([^`=]+)`\s*\|", re.M)


def _entry_names(glossary: str) -> list[str]:
    """Return every spelling the glossary gives an entry row (keyword rows excluded)."""

    return [m.group(1) for m in _ENTRY_ROW.finditer(glossary) if not m.group(1).endswith("*")]


def gone_owners(
    glossary: str, rows: list[dict[str, Any]], keep: Iterable[str] = ()
) -> dict[str, int]:
    """Return the keyword-row owners no dumped callable answers to, with their row counts.

    An owner the glossary still lists as a name (an entry row of its own, as a facade
    taking only ``**kwargs`` has) is not gone, whatever the dump says.

    Parameters
    ----------
    glossary
        Glossary text.
    rows
        Every dumped row (the code's public parameters).
    keep
        Owner spellings to keep whatever the dump says (a facade taking ``**kwargs``).

    Returns
    -------
    dict[str, int]
        Owner spelling to the number of keyword rows naming it.
    """

    aliases = [row["aliases"] for row in rows]
    aliases.append([*_entry_names(glossary), *keep])
    counts: dict[str, int] = {}
    for line in glossary.splitlines():
        match = _KEYWORD_ROW.match(line.strip())
        if not match:
            continue
        for clause in _CLAUSE_SPLIT.split(match.group(2).strip()):
            head = _CLAUSE_HEAD.match(clause)
            run = _OWNER_RUN.match(clause[head.end() - len("keyword of ") :]) if head else None
            if head is None or run is None:
                continue
            for span in _SPAN.findall(run.group(1)):
                owner = span.strip().removesuffix("()")
                if not any(_names(owner, a) for a in aliases):
                    counts[owner] = counts.get(owner, 0) + 1
    return counts


def prune_gone(
    glossary: str, rows: list[dict[str, Any]], keep: Iterable[str] = ()
) -> tuple[str, int, int]:
    """Return the glossary with no keyword row naming a callable the dump no longer has.

    A removed method or class leaves its owner spelling in the rows it used to own;
    ``prune_stale`` cannot see it (no dumped callable answers to the spelling), so this
    pass drops those owners. Measure with ``gone_owners`` first: a spelling the dump
    writes differently would be dropped too.

    Parameters
    ----------
    glossary
        Glossary text.
    rows
        Every dumped row (the code's public parameters).
    keep
        Owner spellings to keep whatever the dump says.

    Returns
    -------
    tuple[str, int, int]
        New text, rows edited, rows removed.
    """

    gone = gone_owners(glossary, rows, keep)
    return _prune(glossary, lambda _name, owner: owner in gone)


def _name(line: str) -> str:
    """Return the keyword a keyword row spells."""

    match = _KEYWORD_ROW.match(line.strip())
    return match.group(1) if match else ""


def _sort_key(line: str) -> str:
    """Return a table row's first backticked spelling without its ``=``."""

    match = re.match(r"^\|\s*`([^`]+)`", line.strip())
    return match.group(1).removesuffix("=") if match else ""


def _rows_to_extend(
    lines: list[str], start: int, end: int, gaps: dict[str, list[dict[str, Any]]]
) -> Iterable[tuple[int, str]]:
    """Yield the one existing row per gapped keyword to extend, preferring the last table."""

    chosen: dict[str, int] = {}
    for index, line in enumerate(lines):
        name = _name(line)
        if name in gaps and (name not in chosen or start <= index < end):
            chosen[name] = index
    return [(index, lines[index]) for index in sorted(chosen.values())]


def main(argv: list[str] | None = None) -> int:
    """Backfill the glossary in place from a dump; print what changed."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dump", type=Path, nargs="?")
    parser.add_argument("--glossary", type=Path, default=Path("docs/reference/glossary.md"))
    parser.add_argument(
        "--prune-records", action="store_true", help="drop record classes from keyword rows"
    )
    parser.add_argument(
        "--prune-stale",
        action="store_true",
        help="drop keywords the dumped code no longer takes from the rows naming them",
    )
    parser.add_argument(
        "--prune-gone",
        action="store_true",
        help="drop owners no dumped callable answers to (a removed method or class)",
    )
    parser.add_argument(
        "--list-gone", action="store_true", help="print the owners --prune-gone would drop"
    )
    parser.add_argument(
        "--keep", action="append", default=[], metavar="OWNER", help="never drop this owner"
    )
    args = parser.parse_args(argv)
    if args.dump is None and not args.prune_records:
        parser.error("a dump or --prune-records is needed")
    text = args.glossary.read_text(encoding="utf-8")
    if args.prune_records:
        text, edited, removed = prune_records(text)
        print(f"pruned record owners from {edited} keyword rows; removed {removed} rows")
    if args.dump is not None:
        rows = json.loads(args.dump.read_text(encoding="utf-8"))
        if args.list_gone:
            for owner, count in sorted(gone_owners(text, rows, args.keep).items()):
                print(f"{count}\t{owner}")
            return 0
        if args.prune_stale:
            text, edited, removed = prune_stale(text, rows)
            print(f"pruned stale keywords from {edited} keyword rows; removed {removed} rows")
        if args.prune_gone:
            text, edited, removed = prune_gone(text, rows, args.keep)
            print(f"pruned gone owners from {edited} keyword rows; removed {removed} rows")
        text, extended, added = backfill(text, rows)
        print(f"extended {extended} keyword rows; added {added} keyword rows")
    args.glossary.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
