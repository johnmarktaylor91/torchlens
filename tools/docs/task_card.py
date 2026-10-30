"""Task-card schema and loader for the Field guide.

A task card is the YAML front matter at the top of a Field guide page. One card
feeds the human page, the landing core block, the door listings, the agent index,
``llms.txt`` and the first-call graders, so its shape is fixed here and checked by
``tests/test_task_card_schema.py``.

Card fields
-----------
``task`` (str)
    What the reader wants to do, in their words.
``one_liner`` (str)
    The result in one sentence.
``canonical`` (str)
    The one call to reach for first.
``avoid`` (str)
    The wrong call people reach for, and why it is wrong.
``tier`` (str)
    One of ``core``, ``kit`` or ``advanced``.
``doors`` (list of str)
    Community doors that list the card: ``mechinterp``, ``neuroai``, ``vision``,
    ``agents``.
``related`` (list of str)
    Slugs of related Field guide pages.
``tags`` (list of str, optional)
    Extra search terms.

Whether each ``canonical`` call executes and resolves to a public name is a
separate check that needs the decided public names; it plugs into
``validate_card`` through the ``resolve_canonical`` hook below.

Run ``python tools/docs/task_card.py PATH`` to validate a card file or folder.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Label used for the how-to pages. Set in one place so a rename is one edit.
SECTION_LABEL = "Field guide"

TIERS = ("core", "kit", "advanced")
DOORS = ("mechinterp", "neuroai", "vision", "agents")
REQUIRED_STR_FIELDS = ("task", "one_liner", "canonical", "avoid", "tier")
REQUIRED_LIST_FIELDS = ("doors", "related")
OPTIONAL_LIST_FIELDS = ("tags",)
ALLOWED_FIELDS = frozenset(REQUIRED_STR_FIELDS + REQUIRED_LIST_FIELDS + OPTIONAL_LIST_FIELDS)
CARD_SUFFIXES = (".md", ".yaml", ".yml")


class TaskCardError(ValueError):
    """Raised when one or more task cards break the schema."""


@dataclass(frozen=True)
class TaskCard:
    """One validated task card.

    Parameters
    ----------
    task, one_liner, canonical, avoid, tier:
        The required string fields (see the module docstring).
    doors, related:
        The required list fields.
    tags:
        Optional extra search terms.
    source:
        The file the card was read from, for error messages.
    """

    task: str
    one_liner: str
    canonical: str
    avoid: str
    tier: str
    doors: tuple[str, ...]
    related: tuple[str, ...]
    tags: tuple[str, ...] = ()
    source: str = field(default="", compare=False)


def _list_problems(name: str, value: Any, allowed: Iterable[str] | None) -> list[str]:
    """Return schema problems for one list field."""

    if not isinstance(value, list):
        return [f"{name} must be a list, got {type(value).__name__}"]
    problems = [
        f"{name}[{i}] must be a non-empty string"
        for i, item in enumerate(value)
        if not isinstance(item, str) or not item.strip()
    ]
    if len(set(map(str, value))) != len(value):
        problems.append(f"{name} has duplicate entries")
    if allowed is not None:
        unknown = sorted({str(v) for v in value} - set(allowed))
        if unknown:
            problems.append(f"{name} has unknown values {unknown}; allowed: {list(allowed)}")
    return problems


def card_problems(
    data: Any,
    resolve_canonical: Callable[[str], str | None] | None = None,
) -> list[str]:
    """List every schema problem in one card mapping.

    Parameters
    ----------
    data:
        The parsed front matter.
    resolve_canonical:
        Optional hook that receives ``canonical`` and returns an error message, or
        ``None`` when the call executes and resolves. The decided-names check
        (launch plan step 8) plugs in here.

    Returns
    -------
    list of str
        Empty when the card is valid.
    """

    if not isinstance(data, dict):
        return [f"card must be a mapping, got {type(data).__name__}"]
    problems = [f"unknown field {key!r}" for key in sorted(set(data) - ALLOWED_FIELDS)]
    for name in REQUIRED_STR_FIELDS:
        value = data.get(name)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{name} must be a non-empty string")
    if isinstance(data.get("tier"), str) and data["tier"] not in TIERS:
        problems.append(f"tier {data['tier']!r} is not one of {list(TIERS)}")
    canonical = data.get("canonical")
    if isinstance(canonical, str) and "\n" in canonical.strip():
        problems.append("canonical must be one call on one line")
    for name in REQUIRED_LIST_FIELDS:
        if name not in data:
            problems.append(f"{name} is required")
    if "doors" in data:
        problems += _list_problems("doors", data["doors"], DOORS)
    if "related" in data:
        problems += _list_problems("related", data["related"], None)
    if "tags" in data:
        problems += _list_problems("tags", data["tags"], None)
    if not problems and resolve_canonical is not None:
        error = resolve_canonical(data["canonical"])
        if error:
            problems.append(f"canonical does not resolve: {error}")
    return problems


def _front_matter(text: str, source: str) -> Any:
    """Return the parsed YAML front matter of a Markdown page."""

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise TaskCardError(f"{source}: no front matter (the page must start with '---')")
    for end, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return yaml.safe_load("\n".join(lines[1:end]))
    raise TaskCardError(f"{source}: front matter is not closed with '---'")


def _raw_cards(path: Path) -> list[Any]:
    """Return the raw card mappings stored in one file."""

    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        return [_front_matter(text, str(path))]
    data = yaml.safe_load(text)
    return data if isinstance(data, list) else [data]


def _card_files(path: Path) -> list[Path]:
    """Return the card files at ``path`` (a file, or a folder searched recursively)."""

    if path.is_file():
        return [path]
    if not path.is_dir():
        raise TaskCardError(f"{path}: no such file or folder")
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in CARD_SUFFIXES)


def load_cards(
    path: str | Path,
    resolve_canonical: Callable[[str], str | None] | None = None,
) -> list[TaskCard]:
    """Load and validate every task card under ``path``.

    Parameters
    ----------
    path:
        A Markdown page with front matter, a YAML file holding one card or a list
        of cards, or a folder searched recursively for both.
    resolve_canonical:
        Optional hook passed to :func:`card_problems`.

    Returns
    -------
    list of TaskCard
        The cards in file order.

    Raises
    ------
    TaskCardError
        Listing every problem in every card, when any card is invalid, when two
        cards share a ``task``, or when no card is found.
    """

    cards: list[TaskCard] = []
    problems: list[str] = []
    for file in _card_files(Path(path)):
        try:
            raw_cards = _raw_cards(file)
        except (TaskCardError, yaml.YAMLError) as error:
            problems.append(f"{file}: {error}")
            continue
        for index, data in enumerate(raw_cards):
            where = f"{file}" if len(raw_cards) == 1 else f"{file}[{index}]"
            found = card_problems(data, resolve_canonical)
            if found:
                problems += [f"{where}: {problem}" for problem in found]
                continue
            cards.append(
                TaskCard(
                    task=data["task"].strip(),
                    one_liner=data["one_liner"].strip(),
                    canonical=data["canonical"].strip(),
                    avoid=data["avoid"].strip(),
                    tier=data["tier"],
                    doors=tuple(data["doors"]),
                    related=tuple(data["related"]),
                    tags=tuple(data.get("tags", ())),
                    source=where,
                )
            )
    seen: dict[str, str] = {}
    for card in cards:
        if card.task in seen:
            problems.append(f"{card.source}: task {card.task!r} also in {seen[card.task]}")
        seen.setdefault(card.task, card.source)
    if not cards and not problems:
        problems.append(f"{path}: no task cards found")
    if problems:
        raise TaskCardError("\n".join(problems))
    return cards


def main(argv: list[str]) -> int:
    """Validate the card paths given on the command line."""

    status = 0
    for arg in argv or ["."]:
        try:
            cards = load_cards(arg)
        except TaskCardError as error:
            print(error)
            status = 1
            continue
        print(f"{arg}: {len(cards)} valid task card(s)")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
