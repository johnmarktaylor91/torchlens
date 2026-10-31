"""The LOCKED lockstep rule, checked for every new, renamed and removed public name.

A change that adds, renames or removes a public name updates, in the same change, the
glossary entry, the agent-reference examples, and the audit notebooks and examples that
use it. This module turns that rule into findings:

- a new or renamed name needs a glossary entry (its head spelling);
- a new root name or namespace export also needs a mention in ``docs/agent-reference/``;
- a renamed or removed name leaves no old spelling in the glossary, the agent
  reference, ``notebooks/audit/`` or ``examples/``.

A ``contextual`` old spelling (``.member``, ``kw=``) counts as residue only when no
head row still uses that leaf, so a member that another class keeps is not flagged, and
only when ``attribution`` cannot show the text belongs to something else: a live head
name at that position, an imported third-party object, a URL, or an assignment target.
A name whose defining module moved but whose public spelling is unchanged (``moved``)
is neither new nor removed; it keeps the glossary-entry obligation only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from name_lockstep import attribution, spelling
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
        ``new``, ``renamed``, ``moved`` or ``removed``.
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
    _imports: dict[tuple[str, str], dict[str, str]] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )

    def imports(self, place: str, path: str) -> dict[str, str]:
        """Return the import bindings of one file, computed once.

        Parameters
        ----------
        place
            Place label.
        path
            The file's label within the place.

        Returns
        -------
        dict[str, str]
            Bound name to top-level package.
        """

        if (place, path) not in self._imports:
            text = dict(self.files[place])[path]
            self._imports[(place, path)] = attribution.imports(text)
        return self._imports[(place, path)]

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


def _stale(
    corpus: Corpus, row: dict[str, Any], head_leaves: set[str], claims: attribution.Claims
) -> list[tuple[str, str]]:
    """Return every place and ``path:line`` that still spells a gone name.

    Parameters
    ----------
    corpus
        Lockstep texts.
    row
        Base row of a renamed or removed name.
    head_leaves
        Leaf names still used by head rows of the same kind.
    claims
        Live head spellings that can own a contextual hit.

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
            lines: list[str] | None = None
            for line, start, end, strength in spelling.find(text, patterns, strengths, prose=False):
                if strength == spelling.CONTEXTUAL:
                    lines = lines if lines is not None else text.splitlines()
                    bound = corpus.imports(place, path)
                    if _explained(row, lines, line, start, end, claims, bound):
                        continue
                found.append((place, f"{path}:{line}"))
    return sorted(set(found))


def _explained(
    row: dict[str, Any],
    lines: list[str],
    number: int,
    start: int,
    end: int,
    claims: attribution.Claims,
    bound: dict[str, str],
) -> bool:
    """Return whether a contextual hit belongs to something other than the gone name.

    Parameters
    ----------
    row
        Base row of the gone name.
    lines
        The file's lines.
    number, start, end
        Hit position: 1-based line, leaf offsets.
    claims
        Live head spellings.
    bound
        The file's import bindings.

    Returns
    -------
    bool
        True when the text is a live name, a third-party attribute, a URL, or (for a
        keyword) an assignment target.
    """

    line = lines[number - 1]
    leaf = str(row["name"])
    if claims.owns(line, leaf, start, end) or attribution.in_url(line, start):
        return True
    if row["kind"] == "member":
        return attribution.third_party(line, start, bound)
    if row["kind"] == "param":
        return attribution.is_assignment(lines, number, start)
    return False


def is_private(row: dict[str, Any]) -> bool:
    """Return whether a row is an underscore keyword, inventoried but not public.

    The inventory lists ``_name=`` keywords so their renames and removals are seen; a
    new one needs no glossary entry and no legibility check.

    Parameters
    ----------
    row
        Inventory row.

    Returns
    -------
    bool
        True for a ``param`` row whose name starts with an underscore.
    """

    return row["kind"] == "param" and str(row["name"]).startswith("_")


def _needs(row: dict[str, Any]) -> list[str]:
    """Return the places a new or renamed name must appear in.

    Parameters
    ----------
    row
        Head row.

    Returns
    -------
    list[str]
        Place labels; none for an underscore keyword.
    """

    if is_private(row):
        return []
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
    claims = attribution.Claims.from_rows(head_rows.values())
    findings: list[Finding] = []
    for row in surface_diff.new:
        findings += _missing(corpus, row, "new")
    for move in surface_diff.moved:
        findings += _missing(corpus, head_rows[move.new_id], "moved")
    for rename in surface_diff.renamed:
        new_row = head_rows[rename.new_id]
        findings += _missing(corpus, new_row, "renamed")
        old_row = {**new_row, "id": rename.old_id, "name": rename.old_name}
        old_row["path"] = _old_path(new_row, rename.new_name, rename.old_name)
        findings += _residue(corpus, old_row, rename.new_id, "renamed", leaves, claims)
    for row in surface_diff.removed:
        findings += _residue(corpus, row, row["id"], "removed", leaves, claims)
    return sorted(findings, key=dataclasses.astuple)


def _old_path(new_row: dict[str, Any], new_name: str, old_name: str) -> str:
    """Return the head row's path with its leaf respelled to the old name.

    A keyword's path is ``name=``, so the leaf sits before the trailing ``=``; every
    other kind ends in the leaf.

    Parameters
    ----------
    new_row
        Head row of a renamed name.
    new_name, old_name
        Head and base leaf spellings.

    Returns
    -------
    str
        The old spelling as a path (``Owner.old``, ``tl.ns.old``, ``old=``).
    """

    path = str(new_row["path"])
    suffix = "=" if new_row["kind"] == "param" else ""
    if not path.endswith(new_name + suffix):
        raise ValueError(f"path {path!r} does not end with {new_name + suffix!r}")
    return path[: len(path) - len(new_name) - len(suffix)] + old_name + suffix


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
    corpus: Corpus,
    row: dict[str, Any],
    report_id: str,
    change: str,
    leaves: dict[str, set[str]],
    claims: attribution.Claims,
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
    claims
        Live head spellings.

    Returns
    -------
    list[Finding]
        Stale-spelling findings.
    """

    return [
        Finding(report_id, change, place, "stale", location, str(row["path"]))
        for place, location in _stale(corpus, row, leaves.get(row["kind"], set()), claims)
    ]
