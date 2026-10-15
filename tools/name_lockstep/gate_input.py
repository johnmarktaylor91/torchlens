"""Turn a surface diff into the input of the new-name legibility check.

Each new or renamed public name becomes one item: its public spelling, kind, signature,
docstring line, a real call-site snippet (5 to 15 lines from tests, examples or the
docs) and the reference meaning a grader compares guesses against (the glossary text
that defines it, else its docstring line).
"""

from __future__ import annotations

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


def _snippets(row: dict[str, Any], corpus: Corpus) -> list[dict[str, Any]]:
    """Return up to three call-site snippets from different files.

    Parameters
    ----------
    row
        Head row.
    corpus
        Texts of the snippet places (``snippets`` key: tests, examples, docs).

    Returns
    -------
    list[dict[str, Any]]
        ``{"path", "line", "text"}`` per snippet; qualified spellings first.
    """

    patterns = spelling.spellings(row)
    found: list[tuple[int, dict[str, Any]]] = []
    for path, text in corpus.files.get("snippets", []):
        strengths = (spelling.QUALIFIED, spelling.CONTEXTUAL)
        hits = spelling.find(text, patterns, strengths, prose=False)
        if not hits:
            continue
        line, _, _, strength = hits[0]
        lines = text.splitlines()
        low, high = max(0, line - 1 - SNIPPET_BEFORE), min(len(lines), line + SNIPPET_AFTER)
        rank = 0 if strength == spelling.QUALIFIED else 1
        found.append((rank, {"path": path, "line": line, "text": "\n".join(lines[low:high])}))
    found.sort(key=lambda item: (item[0], item[1]["path"]))
    return [snippet for _, snippet in found[:MAX_SNIPPETS]]


def _reference(row: dict[str, Any], corpus: Corpus) -> tuple[str, str]:
    """Return the meaning a grader compares guesses against, and where it came from.

    Parameters
    ----------
    row
        Head row.
    corpus
        Lockstep texts.

    Returns
    -------
    tuple[str, str]
        ``(text, source)``; source is ``glossary:<path>:<line>``, ``docstring`` or
        ``none``.
    """

    patterns = spelling.spellings(row)
    strengths = (spelling.QUALIFIED, spelling.CONTEXTUAL, spelling.LEAF)
    for path, text in corpus.files.get("glossary", []):
        hits = spelling.find(text, patterns, strengths, prose=True)
        if hits:
            line = hits[0][0]
            return text.splitlines()[line - 1].strip(), f"glossary:{path}:{line}"
    if row.get("doc_first_line"):
        return str(row["doc_first_line"]), "docstring"
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
    for row, change, old_id in rows:
        snippets = _snippets(row, corpus)
        reference, source = _reference(row, corpus)
        items.append(
            {
                "id": row["id"],
                "name": row["path"],
                "kind": _kind_text(row),
                "signature": row.get("signature"),
                "doc_first_line": row.get("doc_first_line") or None,
                "snippet": snippets[0]["text"] if snippets else None,
                "snippets": snippets,
                "reference": reference,
                "reference_source": source,
                "change": change,
                "old_id": old_id,
            }
        )
    return items
