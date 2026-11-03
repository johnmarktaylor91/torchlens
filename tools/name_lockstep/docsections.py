"""Read the documented meaning of fields and keyword arguments from numpydoc sections.

Dataclass fields, plain attributes and keyword arguments carry no docstring of their
own; their meaning lives in the owner's docstring, under ``Attributes`` or
``Parameters``. This module finds that entry so the legibility check has a doc line
to grade for every kind of name.
"""

from __future__ import annotations

import inspect
import re

#: Longest documented meaning kept per entry (the first paragraph, cut at a word).
MAX_MEANING = 300
FIELD_SECTIONS = ("Attributes", "Parameters")
PARAM_SECTIONS = ("Parameters", "Other Parameters", "Keyword Arguments", "Attributes")
_UNDERLINE = re.compile(r"^\s*-{3,}\s*$")
_ENTRY = re.compile(r"^\*{0,2}(?P<names>[A-Za-z_]\w*(?:\s*,\s*\*{0,2}[A-Za-z_]\w*)*)\s*(?::.*)?$")


def _sections(doc: str) -> dict[str, list[str]]:
    """Split a cleaned docstring into numpydoc sections.

    Parameters
    ----------
    doc
        Docstring as returned by ``inspect.getdoc`` (common indent removed).

    Returns
    -------
    dict[str, list[str]]
        Section title to its body lines; the first section of a title wins.
    """

    lines = doc.splitlines()
    starts = [
        i
        for i in range(len(lines) - 1)
        if lines[i].strip()
        and _UNDERLINE.match(lines[i + 1])
        and len(lines[i + 1].strip()) >= len(lines[i].strip()) // 2
    ]
    found: dict[str, list[str]] = {}
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        found.setdefault(lines[start].strip(), lines[start + 2 : end])
    return found


def _entries(body: list[str]) -> dict[str, str]:
    """Map each documented name in a section body to its first description paragraph.

    Parameters
    ----------
    body
        Section lines after the underline.

    Returns
    -------
    dict[str, str]
        Name to meaning (whitespace joined, at most ``MAX_MEANING`` characters).
    """

    indents = [len(line) - len(line.lstrip()) for line in body if line.strip()]
    if not indents:
        return {}
    base = min(indents)
    out: dict[str, str] = {}
    names: list[str] = []
    text: list[str] = []
    ended = False

    def flush() -> None:
        meaning = _clip(" ".join(text))
        for name in names:
            out.setdefault(name, meaning)

    for line in body:
        indent = len(line) - len(line.lstrip())
        head = _ENTRY.match(line.strip()) if line.strip() and indent == base else None
        if head:
            flush()
            names = [part.strip().lstrip("*") for part in head.group("names").split(",")]
            text, ended = [], False
        elif not line.strip():
            ended = ended or bool(text)
        elif names and not ended and indent > base:
            text.append(line.strip())
    flush()
    return {name: meaning for name, meaning in out.items() if meaning}


def _clip(text: str) -> str:
    """Cut ``text`` to ``MAX_MEANING`` characters at a word boundary."""

    text = " ".join(text.split())
    if len(text) <= MAX_MEANING:
        return text
    return text[:MAX_MEANING].rsplit(" ", 1)[0] + " ..."


def documented(obj: object, name: str, sections: tuple[str, ...]) -> tuple[str, str]:
    """Return the documented meaning of ``name`` in ``obj``'s docstring sections.

    Parameters
    ----------
    obj
        A class or callable whose docstring may document ``name``.
    name
        Field, attribute or keyword name.
    sections
        Section titles to search, in order.

    Returns
    -------
    tuple[str, str]
        ``(meaning, section)``, or ``("", "")`` when no section documents ``name``.
    """

    found = _sections(inspect.getdoc(obj) or "")
    for title in sections:
        meaning = _entries(found.get(title, [])).get(name)
        if meaning:
            return meaning, title
    return "", ""


def field_meaning(owner: type, name: str) -> tuple[str, str]:
    """Return a class member's meaning from its own or a base class's docstring.

    Parameters
    ----------
    owner
        The defining class.
    name
        Field or attribute name.

    Returns
    -------
    tuple[str, str]
        ``(meaning, source)`` with source ``<Class> <Section>``, or ``("", "")``.
    """

    for cls in inspect.getmro(owner):
        meaning, title = documented(cls, name, FIELD_SECTIONS)
        if meaning:
            return meaning, f"{cls.__qualname__} {title}"
    return "", ""


def param_meaning(target: object, name: str) -> tuple[str, str]:
    """Return a keyword argument's meaning from the callable that takes it.

    A class documents its constructor keywords in its own docstring (``Parameters``,
    else ``Attributes`` for dataclasses) or in ``__init__``'s.

    Parameters
    ----------
    target
        Function, method or class.
    name
        Keyword name.

    Returns
    -------
    tuple[str, str]
        ``(meaning, section)``, or ``("", "")``.
    """

    meaning, title = documented(target, name, PARAM_SECTIONS)
    if not meaning and inspect.isclass(target) and "__init__" in vars(target):
        meaning, title = documented(vars(target)["__init__"], name, PARAM_SECTIONS)
    return meaning, title
