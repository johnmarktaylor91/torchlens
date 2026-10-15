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


def _context_terms(row: dict[str, Any]) -> list[str]:
    """Return the words that place a contextual hit in this name's own context.

    A member's context is its owner class; a keyword's is a callable that takes it.

    Parameters
    ----------
    row
        Head row.

    Returns
    -------
    list[str]
        Owner class names (members) or callable leaf names (keywords); empty otherwise.
    """

    if row["kind"] == "member":
        return sorted({*row.get("owner_names", ()), str(row["path"]).split(".")[0]})
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
        ``{"path", "line", "text"}`` per snippet; qualified spellings first, then
        contextual spellings near the owner class (members) or a callable that takes the
        keyword (keywords), then other contextual spellings.
    """

    patterns = spelling.spellings(row)
    terms = _context_terms(row)
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


def _reference(row: dict[str, Any], corpus: Corpus) -> tuple[str, str]:
    """Return the meaning a grader compares guesses against, and where it came from.

    The glossary line with the strongest spelling wins. For a member or keyword a
    leaf-only glossary hit (the bare word, which index tables list for many objects)
    ranks below the documented meaning read from the owner's docstring sections.

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

    patterns = spelling.spellings(row)
    best: tuple[int, str, str] | None = None
    for path, text in corpus.files.get("glossary", []):
        for line, _, _, strength in spelling.find(text, patterns, _STRENGTH_ORDER, prose=True):
            rank = _STRENGTH_ORDER.index(strength)
            if best is None or rank < best[0]:
                best = (rank, text.splitlines()[line - 1].strip(), f"glossary:{path}:{line}")
    documented = str(row.get("doc_first_line") or "")
    doc_source = row.get("doc_source") or "docstring"
    doc_source = doc_source if doc_source == "docstring" else f"doc:{doc_source}"
    leaf_only = best is not None and best[0] == _STRENGTH_ORDER.index(spelling.LEAF)
    if documented and row["kind"] in ("member", "param") and (best is None or leaf_only):
        return documented, doc_source
    if best is not None:
        return best[1], best[2]
    if documented:
        return documented, doc_source
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
