"""Turn a surface diff into the input of the new-name legibility check.

Each new or renamed public name becomes one item: its public spelling, kind, signature,
documented meaning (its docstring line, or for a field or keyword the entry in the
owner's numpydoc ``Attributes`` or ``Parameters`` section), a real call-site snippet (5
to 15 lines from tests, examples or the docs) and the reference meaning a grader compares
guesses against (the glossary text that defines it, else its documented meaning).
"""

from __future__ import annotations

import re
from typing import Any

from name_lockstep import spelling
from name_lockstep.check import Corpus
from name_lockstep.diff import SurfaceDiff

SNIPPET_BEFORE = 6
SNIPPET_AFTER = 6
MAX_SNIPPETS = 3
_KIND_TEXT = {
    "root": "top-level `tl.` name",
    "export": "namespace export",
    "member": "member of a TorchLens object",
    "param": "keyword argument",
}


def _kind_text(row: dict[str, Any]) -> str:
    """Describe a row's kind the way the legibility primer does.

    Parameters
    ----------
    row
        Head row.

    Returns
    -------
    str
        For example ``function, top-level `tl.` name``.
    """

    return f"{row.get('subkind') or row['kind']}, {_KIND_TEXT.get(row['kind'], row['kind'])}"


_RETURNS = re.compile(r"->\s*['\"]?(?:[A-Za-z_][\w]*\.)*([A-Za-z_]\w*)['\"]?\s*$")


def producers(head_rows: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    """Map each class name to the leaf names of the functions and methods that return it.

    Parameters
    ----------
    head_rows
        Head inventory keyed on id.

    Returns
    -------
    dict[str, set[str]]
        Class name to producer leaf names, read from each signature's return annotation.
    """

    found: dict[str, set[str]] = {}
    for row in head_rows.values():
        match = _RETURNS.search(str(row.get("signature") or ""))
        if match and row["kind"] != "param":
            found.setdefault(match.group(1), set()).add(str(row["path"]).rsplit(".", 1)[-1])
    return found


def _context_terms(row: dict[str, Any], made_by: dict[str, set[str]] | None = None) -> list[str]:
    """Return the words that place a contextual hit in this name's own context.

    A member's context is its owner class and the calls that return it; a keyword's is
    a callable that takes it.

    Parameters
    ----------
    row
        Head row.
    made_by
        ``producers`` of the head inventory (optional).

    Returns
    -------
    list[str]
        Owner class and producer names (members) or callable leaf names (keywords);
        empty otherwise.
    """

    made_by = made_by or {}
    if row["kind"] == "member":
        owners = {*row.get("owner_names", ()), str(row["path"]).split(".")[0]}
        return sorted(owners.union(*(made_by.get(owner, set()) for owner in owners)))
    if row["kind"] == "param":
        return sorted({str(use).rsplit(".", 1)[-1] for use in row.get("uses", ())})
    return []


def _best_hit(
    hits: list[tuple[int, int, int, str]], lines: list[str], terms: list[str]
) -> tuple[int, int]:
    """Pick a file's best hit: qualified, then contextual near a context term, then first.

    Parameters
    ----------
    hits
        ``spelling.find`` hits for one file.
    lines
        The file's lines.
    terms
        Context terms from ``_context_terms``.

    Returns
    -------
    tuple[int, int]
        ``(rank, line)``: rank 0 qualified, 1 contextual in context, 2 other contextual.
    """

    pattern = re.compile(rf"\b(?:{'|'.join(map(re.escape, terms))})\b") if terms else None
    context = {n for n, text in enumerate(lines, start=1) if pattern and pattern.search(text)}
    ranked = []
    for line, _, _, strength in hits:
        near = any(abs(line - n) <= SNIPPET_BEFORE + SNIPPET_AFTER for n in context)
        rank = 0 if strength == spelling.QUALIFIED else (1 if near else 2)
        ranked.append((rank, line))
    return min(ranked)


def _snippets(
    row: dict[str, Any], corpus: Corpus, made_by: dict[str, set[str]] | None = None
) -> list[dict[str, Any]]:
    """Return up to three call-site snippets from different files.

    Parameters
    ----------
    row
        Head row.
    corpus
        Texts of the snippet places (``snippets`` key: tests, examples, docs).
    made_by
        ``producers`` of the head inventory (optional).

    Returns
    -------
    list[dict[str, Any]]
        ``{"path", "line", "text"}`` per snippet; qualified spellings first, then
        contextual spellings near the owner class or a call that returns it (members) or a
        callable that takes the
        keyword (keywords), then other contextual spellings.
    """

    patterns = spelling.spellings(row)
    terms = _context_terms(row, made_by)
    found: list[tuple[int, dict[str, Any]]] = []
    for path, text in corpus.files.get("snippets", []):
        strengths = (spelling.QUALIFIED, spelling.CONTEXTUAL)
        hits = spelling.find(text, patterns, strengths, prose=False)
        if not hits:
            continue
        lines = text.splitlines()
        rank, line = _best_hit(hits, lines, terms)
        low, high = max(0, line - 1 - SNIPPET_BEFORE), min(len(lines), line + SNIPPET_AFTER)
        found.append((rank, {"path": path, "line": line, "text": "\n".join(lines[low:high])}))
    found.sort(key=lambda item: (item[0], item[1]["path"]))
    return [snippet for _, snippet in found[:MAX_SNIPPETS]]


_STRENGTH_ORDER = (spelling.QUALIFIED, spelling.CONTEXTUAL, spelling.LEAF)
#: Glossary candidate kinds, best first: a per-path index entry keyed on the row id, a
#: definition-list term, a prose line or a definition table row, a spelling-index row.
_KIND_ORDER = ("id", "term", "prose", "index")
_TICKED_LIST = re.compile(r"`[^`]+`\s*,\s*`[^`]+`")


def _definition(lines: list[str], index: int) -> str:
    """Return the ``: definition`` paragraph that follows a ``**Term**`` line, or ``""``.

    Parameters
    ----------
    lines
        Glossary lines.
    index
        Zero-based index of the term line.

    Returns
    -------
    str
        The definition text, whitespace joined, without its leading ``: ``.
    """

    following = [i for i in range(index + 1, min(len(lines), index + 4)) if lines[i].strip()]
    if not following or not lines[following[0]].startswith(": "):
        return ""
    body = [lines[following[0]][2:].strip()]
    for line in lines[following[0] + 1 :]:
        if not line.strip() or not line.startswith((" ", "\t")):
            break
        body.append(line.strip())
    return " ".join(body)


def _candidate(row: dict[str, Any], lines: list[str], number: int) -> tuple[str, str]:
    """Classify one glossary hit and return ``(kind, definition text)``.

    Parameters
    ----------
    row
        Head row.
    lines
        Glossary lines.
    number
        One-based line number of the hit.

    Returns
    -------
    tuple[str, str]
        Kind from ``_KIND_ORDER`` and the text a grader should compare against.
    """

    line = lines[number - 1].strip()
    if f"id `{row['id']}`" in line and "): " in line:
        return "id", line.split("): ", 1)[1]
    if line.startswith("**"):
        definition = _definition(lines, number - 1)
        if definition:
            return "term", definition
    if line.startswith("|"):
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if any(_TICKED_LIST.search(cell) for cell in cells):
            return "index", line
    return "prose", line


def _glossary_candidates(row: dict[str, Any], corpus: Corpus) -> list[tuple[int, int, str, str]]:
    """Return ``(kind rank, strength rank, text, source)`` for every glossary hit, best first.

    Parameters
    ----------
    row
        Head row.
    corpus
        Lockstep texts.

    Returns
    -------
    list[tuple[int, int, str, str]]
        Sorted candidates; source is ``glossary:<path>:<line>``.
    """

    patterns = spelling.spellings(row)
    found = []
    for path, text in corpus.files.get("glossary", []):
        lines = text.splitlines()
        for number, _, _, strength in spelling.find(text, patterns, _STRENGTH_ORDER, prose=True):
            kind, definition = _candidate(row, lines, number)
            found.append(
                (
                    _KIND_ORDER.index(kind),
                    _STRENGTH_ORDER.index(strength),
                    definition,
                    f"glossary:{path}:{number}",
                )
            )
    return sorted(found, key=lambda item: (item[0], item[1], item[3]))


def _reference(row: dict[str, Any], corpus: Corpus) -> tuple[str, str]:
    """Return the meaning a grader compares guesses against, and where it came from.

    Only definition text counts: a per-path index entry for this exact row id, else a
    ``**Term**`` entry's ``: definition`` paragraph or a prose or table line naming it;
    a spelling-index row (a cell listing several backticked spellings) is a last resort
    below the documented meaning. For a member or keyword a bare-word (leaf) hit names
    many objects, so it also ranks below the documented meaning read from the owner's
    docstring sections.

    Parameters
    ----------
    row
        Head row.
    corpus
        Lockstep texts.

    Returns
    -------
    tuple[str, str]
        ``(text, source)``; source is ``glossary:<path>:<line>``, ``docstring`` (the
        name's own docstring), ``doc:<owner> <Section>`` or ``none``.
    """

    candidates = _glossary_candidates(row, corpus)
    documented = str(row.get("doc_first_line") or "")
    doc_source = row.get("doc_source") or "docstring"
    doc_source = doc_source if doc_source == "docstring" else f"doc:{doc_source}"
    index_rank, leaf_rank = _KIND_ORDER.index("index"), _STRENGTH_ORDER.index(spelling.LEAF)
    ambiguous_leaf = row["kind"] in ("member", "param")

    def strong(item: tuple[int, int, str, str]) -> bool:
        return item[0] == 0 or (
            item[0] < index_rank and not (ambiguous_leaf and item[1] == leaf_rank)
        )

    for item in candidates:
        if strong(item):
            return item[2], item[3]
    if documented:
        return documented, doc_source
    for item in candidates:
        if item[0] < index_rank:
            return item[2], item[3]
    if candidates:
        return candidates[0][2], candidates[0][3]
    return "", "none"


def gate_items(surface_diff: SurfaceDiff, head_rows: dict[str, dict[str, Any]], corpus: Corpus):
    """Return one legibility-check item per new or renamed name.

    Parameters
    ----------
    surface_diff
        New, removed and renamed names.
    head_rows
        Head inventory keyed on id.
    corpus
        Lockstep texts plus a ``snippets`` place.

    Returns
    -------
    list[dict[str, Any]]
        Items with ``id``, ``name``, ``kind``, ``signature``, ``doc_first_line``,
        ``snippet``, ``snippets``, ``reference``, ``reference_source`` and ``change``.
    """

    rows = [(row, "new", None) for row in surface_diff.new]
    rows += [(head_rows[r.new_id], "renamed", r.old_id) for r in surface_diff.renamed]
    items = []
    made_by = producers(head_rows)
    for row, change, old_id in rows:
        snippets = _snippets(row, corpus, made_by)
        reference, source = _reference(row, corpus)
        items.append(
            {
                "id": row["id"],
                "name": row["path"],
                "kind": _kind_text(row),
                "signature": row.get("signature"),
                "doc_first_line": row.get("doc_first_line") or None,
                "doc_source": row.get("doc_source")
                or ("docstring" if row.get("doc_first_line") else None),
                "snippet": snippets[0]["text"] if snippets else None,
                "snippets": snippets,
                "reference": reference,
                "reference_source": source,
                "change": change,
                "old_id": old_id,
            }
        )
    return items
