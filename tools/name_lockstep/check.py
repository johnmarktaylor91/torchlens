"""The LOCKED lockstep rule, checked for every new, renamed and removed public name.

A change that adds, renames or removes a public name updates, in the same change, the
glossary entry, the agent-reference examples, and the audit notebooks and examples that
use it. This module turns that rule into findings:

- a new or renamed name needs a glossary entry (its head spelling);
- a new root name or namespace export also needs a mention in ``docs/agent-reference/``;
- a renamed or removed name leaves no old spelling in the glossary, the agent
  reference, ``notebooks/audit/`` or ``examples/``.

A ``contextual`` old spelling (``.member``, ``kw=``) counts as residue only when no
head row still uses that leaf, so a member that another class keeps is not flagged.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from name_lockstep import spelling
from name_lockstep.diff import SurfaceDiff

DEFAULT_GLOSSARY = ("docs/reference/glossary.md",)
#: Lockstep places, as (label, directory, suffixes).
PLACES = (
    ("agent-reference", "docs/agent-reference", (".md",)),
    ("notebooks/audit", "notebooks/audit", (".ipynb", ".py", ".md")),
    ("examples", "examples", (".py", ".md", ".ipynb")),
)
#: Where call-site snippets for the legibility check come from.
SNIPPET_PLACES = (("tests", (".py",)), ("examples", (".py", ".ipynb")), ("docs", (".md",)))
_AGENT_REFERENCE_KINDS = ("root", "export")


@dataclasses.dataclass(frozen=True)
class Finding:
    """One place the lockstep rule is not met.

    Attributes
    ----------
    row_id
        Stable id of the name (the head id for new and renamed names).
    change
        ``new``, ``renamed`` or ``removed``.
    place
        ``glossary``, ``agent-reference``, ``notebooks/audit`` or ``examples``.
    problem
        ``missing`` (no entry or mention) or ``stale`` (old spelling remains).
    location
        ``path:line`` for stale spellings, else the searched place.
    spelling
        The spelling looked for.
    """

    row_id: str
    change: str
    place: str
    problem: str
    location: str
    spelling: str

    def render(self) -> str:
        """Return a one-line description.

        Returns
        -------
        str
            Human-readable finding.
        """

        verb = "has no entry for" if self.problem == "missing" else "still spells"
        return (
            f"{self.change} {self.row_id}: {self.place} {verb} {self.spelling!r} ({self.location})"
        )


@dataclasses.dataclass
class Corpus:
    """The text of every lockstep place, read once.

    Attributes
    ----------
    root
        Repository root.
    files
        Place label to ``[(relative path, text)]``.
    """

    root: Path
    files: dict[str, list[tuple[str, str]]]

    @classmethod
    def read(cls, root: Path, glossaries: Iterable[Path], snippets: bool = False) -> Corpus:
        """Read the glossaries and every file under the lockstep places.

        Parameters
        ----------
        root
            Repository root.
        glossaries
            Glossary files (absolute, or relative to ``root``).
        snippets
            Also read the snippet places into a ``snippets`` entry.

        Returns
        -------
        Corpus
            The texts.
        """

        files: dict[str, list[tuple[str, str]]] = {"glossary": []}
        for glossary in glossaries:
            path = glossary if glossary.is_absolute() else root / glossary
            files["glossary"].append((_label(path, root), path.read_text(encoding="utf-8")))
        for label, directory, suffixes in PLACES:
            base = root / directory
            paths = sorted(p for p in base.rglob("*") if p.suffix in suffixes and p.is_file())
            files[label] = [(_label(p, root), p.read_text(encoding="utf-8")) for p in paths]
        if snippets:
            glossary_labels = {label for label, _ in files["glossary"]}
            files["snippets"] = [
                (_label(p, root), p.read_text(encoding="utf-8"))
                for directory, suffixes in SNIPPET_PLACES
                for p in sorted((root / directory).rglob("*"))
                if p.suffix in suffixes and p.is_file() and _label(p, root) not in glossary_labels
            ]
        return cls(root=root, files=files)


def _label(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` when it is inside it.

    Parameters
    ----------
    path
        A file path.
    root
        Repository root.

    Returns
    -------
    str
        Relative path, or the absolute path for files outside the repository.
    """

    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _present(corpus: Corpus, place: str, row: dict[str, Any]) -> bool:
    """Return whether a place spells the row's current name.

    Parameters
    ----------
    corpus
        Lockstep texts.
    place
        Place label.
    row
        Head row.

    Returns
    -------
    bool
        True when any qualified or contextual spelling, or an emphasised leaf, appears.
    """

    patterns = spelling.spellings(row)
    strengths = (spelling.QUALIFIED, spelling.CONTEXTUAL, spelling.LEAF)
    return any(
        spelling.find(text, patterns, strengths, prose=True) for _, text in corpus.files[place]
    )


def _stale(corpus: Corpus, row: dict[str, Any], head_leaves: set[str]) -> list[tuple[str, str]]:
    """Return every place and ``path:line`` that still spells a gone name.

    Parameters
    ----------
    corpus
        Lockstep texts.
    row
        Base row of a renamed or removed name.
    head_leaves
        Leaf names still used by head rows of the same kind.

    Returns
    -------
    list[tuple[str, str]]
        ``(place, "path:line")`` pairs.
    """

    strengths: tuple[str, ...] = (spelling.QUALIFIED,)
    if row["name"] not in head_leaves:
        strengths += (spelling.CONTEXTUAL,)
    patterns = spelling.spellings(row)
    found: list[tuple[str, str]] = []
    for place in ("glossary", *(label for label, _, _ in PLACES)):
        for path, text in corpus.files.get(place, []):
            for line, *_ in spelling.find(text, patterns, strengths, prose=False):
                found.append((place, f"{path}:{line}"))
    return sorted(set(found))


def _needs(row: dict[str, Any]) -> list[str]:
    """Return the places a new or renamed name must appear in.

    Parameters
    ----------
    row
        Head row.

    Returns
    -------
    list[str]
        Place labels.
    """

    places = ["glossary"]
    if row["kind"] in _AGENT_REFERENCE_KINDS:
        places.append("agent-reference")
    return places


def check_lockstep(
    surface_diff: SurfaceDiff, corpus: Corpus, head_rows: dict[str, dict[str, Any]]
) -> list[Finding]:
    """Check the lockstep rule over a surface diff.

    Parameters
    ----------
    surface_diff
        New, removed and renamed names.
    corpus
        Lockstep texts at the head side.
    head_rows
        Head inventory keyed on id.

    Returns
    -------
    list[Finding]
        Every unmet obligation, sorted.
    """

    leaves: dict[str, set[str]] = {}
    for row in head_rows.values():
        leaves.setdefault(row["kind"], set()).add(row["name"])
    findings: list[Finding] = []
    for row in surface_diff.new:
        findings += _missing(corpus, row, "new")
    for rename in surface_diff.renamed:
        new_row = head_rows[rename.new_id]
        findings += _missing(corpus, new_row, "renamed")
        old_row = {**new_row, "id": rename.old_id, "name": rename.old_name}
        old_row["path"] = str(new_row["path"])[: -len(rename.new_name)] + rename.old_name
        findings += _residue(corpus, old_row, rename.new_id, "renamed", leaves)
    for row in surface_diff.removed:
        findings += _residue(corpus, row, row["id"], "removed", leaves)
    return sorted(findings, key=dataclasses.astuple)


def _missing(corpus: Corpus, row: dict[str, Any], change: str) -> list[Finding]:
    """Return the places that lack the row's current spelling.

    Parameters
    ----------
    corpus
        Lockstep texts.
    row
        Head row.
    change
        ``new`` or ``renamed``.

    Returns
    -------
    list[Finding]
        One finding per missing place.
    """

    return [
        Finding(row["id"], change, place, "missing", place, str(row["path"]))
        for place in _needs(row)
        if not _present(corpus, place, row)
    ]


def _residue(
    corpus: Corpus, row: dict[str, Any], report_id: str, change: str, leaves: dict[str, set[str]]
) -> list[Finding]:
    """Return a finding per place and line that still spells a gone name.

    Parameters
    ----------
    corpus
        Lockstep texts.
    row
        A row carrying the old spelling.
    report_id
        Id to report the finding under.
    change
        ``renamed`` or ``removed``.
    leaves
        Head leaf names per kind.

    Returns
    -------
    list[Finding]
        Stale-spelling findings.
    """

    return [
        Finding(report_id, change, place, "stale", location, str(row["path"]))
        for place, location in _stale(corpus, row, leaves.get(row["kind"], set()))
    ]
