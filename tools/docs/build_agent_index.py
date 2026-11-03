"""Generate the agent index, the skill router, ``llms.txt`` and the capabilities list.

Every output comes from task cards, so humans, agents and the first-call tests read one source.
A card is either an entry of ``docs/field-guide/task_cards.yaml`` or the YAML front matter of a
Field guide page (``docs/field-guide/*.md``, slug = file name). The script needs no docs build
tool, so it runs before any site build.

Card fields: ``id`` (cards file only; a front-matter card's id is its page slug), ``task``,
``one_liner``, ``canonical`` (one call, or ``none`` when TorchLens is the wrong tool),
``avoid``, ``tier`` (``core``, ``kit`` or ``advanced``), ``doors`` (from ``mechinterp``,
``neuroai``, ``vision``, ``agents``), ``related`` (card ids) and, in the cards file, ``page``
(repository-relative path of the page that shows the task). Optional: ``tags``, and the
first-call fields ``prompt``, ``fixture``, ``accept``, ``avoid_calls`` and ``hard_fail``
(read by ``tools/agent_evals/first_call.py``).

Outputs:

``torchlens/_docs/AGENT_INDEX.md``
    At most 8 KB, shipped in the wheel: the rules agents get wrong, the core calls, a
    task-to-call table with each card's ``avoid`` line, and links to the pages.
    ``torchlens agent init`` points a project's instruction file at the installed copy.
``skills/torchlens/ROUTER.md``
    The same text, beside the installable skill's ``SKILL.md``.
``docs/reference/agent_capabilities.md``
    The capabilities list: each Silico capability name mapped to the cards that cover it.
``tools/docs/generated/llms.txt``
    The llms.txt index (one link line per card).
``llms-full.txt`` (only with ``--full PATH``; untracked)
    The full Markdown of every linked page, for a docs site build. It changes with every
    page edit, so it is built where it is published, not committed.

Usage::

    python tools/docs/build_agent_index.py           # write every tracked output
    python tools/docs/build_agent_index.py --check   # exit 1 if any tracked output is stale
    python tools/docs/build_agent_index.py --full site/llms-full.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
CARDS_FILE = REPO / "docs" / "field-guide" / "task_cards.yaml"
CARDS_DIR = REPO / "docs" / "field-guide"
#: Link base for pages. Switch to the docs site's versioned root once it is live.
DEFAULT_LINK_BASE = "https://github.com/johnmarktaylor91/torchlens/blob/main/"
INDEX_LIMIT_BYTES = 8 * 1024

INDEX_PATH = Path("torchlens/_docs/AGENT_INDEX.md")
ROUTER_PATH = Path("skills/torchlens/ROUTER.md")
CAPABILITIES_PATH = Path("docs/reference/agent_capabilities.md")
LLMS_PATH = Path("tools/docs/generated/llms.txt")

TIERS: tuple[str, ...] = ("core", "kit", "advanced")
DOORS: frozenset[str] = frozenset({"mechinterp", "neuroai", "vision", "agents"})
FIXTURES: frozenset[str] = frozenset({"vision", "lm", "none"})
REQUIRED: tuple[str, ...] = ("task", "one_liner", "canonical", "avoid", "tier", "doors", "related")
FIRST_CALL_FIELDS: tuple[str, ...] = ("prompt", "fixture", "accept", "avoid_calls", "hard_fail")
OPTIONAL: tuple[str, ...] = ("tags", *FIRST_CALL_FIELDS)
#: The canonical value that means "TorchLens is the wrong tool for this task".
NO_CALL = "none"


class CardError(ValueError):
    """A task card, or the cards file, is malformed."""


@dataclass(frozen=True)
class Card:
    """One task card and the page that shows it."""

    slug: str
    task: str
    one_liner: str
    canonical: str
    avoid: str
    tier: str
    doors: tuple[str, ...]
    related: tuple[str, ...]
    tags: tuple[str, ...]
    page: str
    prompt: str = ""
    fixture: str = "none"
    accept: tuple[str, ...] = ()
    avoid_calls: tuple[str, ...] = ()
    hard_fail: bool = False


@dataclass(frozen=True)
class Capability:
    """One Silico capability name and the cards that cover it."""

    name: str
    cards: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class CardSet:
    """Everything the cards file and the carded pages declare."""

    rules: tuple[str, ...]
    cards: tuple[Card, ...]
    capabilities: tuple[Capability, ...]

    def by_slug(self) -> dict[str, Card]:
        """Return the cards keyed by slug."""
        return {card.slug: card for card in self.cards}


def split_front_matter(text: str) -> tuple[dict[str, Any] | None, str]:
    """Split a page into its YAML front matter (or ``None``) and its body.

    Parameters
    ----------
    text : str
        The page source.

    Returns
    -------
    tuple[dict[str, Any] | None, str]
        The parsed front matter, and the body after it.

    Examples
    --------
    >>> split_front_matter("---\\ntask: t\\n---\\n# Page\\n")
    ({'task': 't'}, '# Page\\n')
    """
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return None, text
    meta = yaml.safe_load(text[4:end]) or {}
    if not isinstance(meta, dict):
        return None, text
    return meta, text[end + 5 :].lstrip("\n")


def _string_list(meta: dict[str, Any], key: str) -> bool:
    value = meta.get(key, [])
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def card_problems(meta: dict[str, Any]) -> list[str]:
    """Return every field-contract problem of one card's fields.

    Parameters
    ----------
    meta : dict[str, Any]
        The card's fields, without ``id`` and ``page``.

    Returns
    -------
    list[str]
        One message per problem; empty when the card is well formed.

    Examples
    --------
    >>> card_problems({"task": "t"})[0].startswith("missing")
    True
    """
    missing = [key for key in REQUIRED if key not in meta]
    unknown = sorted(set(meta) - set(REQUIRED) - set(OPTIONAL))
    if missing or unknown:
        return [f"missing {missing}, unknown {unknown}"]
    problems = [
        f"{key} must be a non-empty string"
        for key in ("task", "one_liner", "canonical", "avoid")
        if not isinstance(meta[key], str) or not meta[key].strip()
    ]
    if meta["tier"] not in TIERS:
        problems.append(f"tier must be one of {TIERS}")
    problems += [
        f"{key} must be a list of strings"
        for key in ("doors", "related", "tags", "accept", "avoid_calls")
        if not _string_list(meta, key)
    ]
    if not problems and not set(meta["doors"]) <= DOORS:
        problems.append(f"unknown doors {sorted(set(meta['doors']) - DOORS)}")
    if meta.get("fixture", "none") not in FIXTURES:
        problems.append(f"fixture must be one of {sorted(FIXTURES)}")
    if not isinstance(meta.get("prompt", ""), str):
        problems.append("prompt must be a string")
    if not isinstance(meta.get("hard_fail", False), bool):
        problems.append("hard_fail must be true or false")
    return problems


def card_from_meta(slug: str, page: str, meta: dict[str, Any]) -> Card:
    """Build a card from its fields, checking the field contract.

    Parameters
    ----------
    slug : str
        The card id.
    page : str
        Repository-relative path of the page that shows the task.
    meta : dict[str, Any]
        The card's fields, without ``id`` and ``page``.

    Returns
    -------
    Card
        The checked card.

    Raises
    ------
    CardError
        On a missing, unknown or ill-typed field.

    Examples
    --------
    >>> card = card_from_meta("c", "docs/quickstart.md", {
    ...     "task": "T", "one_liner": "O", "canonical": "tl.trace(model, x)",
    ...     "avoid": "A", "tier": "core", "doors": [], "related": []})
    >>> card.canonical
    'tl.trace(model, x)'
    """
    problems = card_problems(meta)
    if problems:
        raise CardError(f"{slug}: " + "; ".join(problems))
    return Card(
        slug=slug,
        task=meta["task"].strip(),
        one_liner=meta["one_liner"].strip(),
        canonical=meta["canonical"].strip(),
        avoid=meta["avoid"].strip(),
        tier=meta["tier"],
        doors=tuple(meta["doors"]),
        related=tuple(meta["related"]),
        tags=tuple(meta.get("tags", [])),
        page=page,
        prompt=str(meta.get("prompt", "")).strip(),
        fixture=meta.get("fixture", "none"),
        accept=tuple(meta.get("accept", [])),
        avoid_calls=tuple(meta.get("avoid_calls", [])),
        hard_fail=bool(meta.get("hard_fail", False)),
    )


def _file_cards(raw: list[Any], repo: Path) -> list[Card]:
    cards = []
    for entry in raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise CardError(f"every card needs a string id: {entry!r}")
        fields = dict(entry)
        slug, page = fields.pop("id"), fields.pop("page", None)
        if not isinstance(page, str) or not (repo / page.split("#")[0]).is_file():
            raise CardError(f"{slug}: page {page!r} is not a file in the repository")
        cards.append(card_from_meta(slug, page, fields))
    return cards


def _page_cards(cards_dir: Path, repo: Path) -> list[Card]:
    cards = []
    for page in sorted(cards_dir.glob("*.md")):
        meta, _ = split_front_matter(page.read_text("utf-8"))
        if meta is None or "task" not in meta:
            continue
        cards.append(card_from_meta(page.stem, page.relative_to(repo).as_posix(), meta))
    return cards


def _capabilities(raw: list[Any], slugs: set[str]) -> tuple[Capability, ...]:
    rows = []
    for entry in raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise CardError(f"every capability needs a string name: {entry!r}")
        cards = tuple(entry.get("cards", []))
        stray = sorted(set(cards) - slugs)
        if stray:
            raise CardError(f"capability {entry['name']}: unknown cards {stray}")
        note = str(entry.get("note", "")).strip()
        if not cards and not note:
            raise CardError(f"capability {entry['name']}: no cards and no note")
        rows.append(Capability(entry["name"], cards, note))
    return tuple(rows)


def load_card_set(
    cards_file: Path = CARDS_FILE, cards_dir: Path = CARDS_DIR, repo: Path = REPO
) -> CardSet:
    """Read the cards file and every carded page, ordered by tier, then file order.

    Parameters
    ----------
    cards_file : Path
        The YAML cards file.
    cards_dir : Path
        Directory whose pages may carry a card as front matter.
    repo : Path
        Repository root that ``page`` paths are relative to.

    Returns
    -------
    CardSet
        The rules, cards and capabilities.

    Raises
    ------
    CardError
        On a malformed card, a duplicate id, or a ``related`` id that names no card.

    Examples
    --------
    >>> card_set = load_card_set()
    >>> card_set.by_slug()["capture"].canonical
    'trace = tl.trace(model, x)'
    """
    data = yaml.safe_load(cards_file.read_text("utf-8")) or {}
    cards = _file_cards(data.get("cards", []), repo) + _page_cards(cards_dir, repo)
    slugs = [card.slug for card in cards]
    duplicates = sorted({slug for slug in slugs if slugs.count(slug) > 1})
    if duplicates:
        raise CardError(f"duplicate card ids: {duplicates}")
    for card in cards:
        stray = sorted(set(card.related) - set(slugs))
        if stray:
            raise CardError(f"{card.slug}: related names no card: {stray}")
    ordered = sorted(cards, key=lambda c: TIERS.index(c.tier))
    rules = tuple(str(rule).strip() for rule in data.get("rules", []))
    return CardSet(rules, tuple(ordered), _capabilities(data.get("capabilities", []), set(slugs)))


def _project_summary(repo: Path) -> str:
    text = (repo / "pyproject.toml").read_text("utf-8")
    match = re.search(r'^description\s*=\s*"(.*)"\s*$', text, re.MULTILINE)
    if match is None:
        raise CardError("pyproject.toml has no one-line project description")
    return match.group(1).strip()


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _call_cell(card: Card) -> str:
    if card.canonical == NO_CALL:
        return "none: not a TorchLens task"
    return f"`{_cell(card.canonical)}`"


def render_index(card_set: CardSet, link_base: str, summary: str) -> str:
    """Render ``AGENT_INDEX.md`` (also the skill's ``ROUTER.md``).

    Parameters
    ----------
    card_set : CardSet
        The loaded cards.
    link_base : str
        URL prefix that a repository-relative page path is appended to.
    summary : str
        The one-line project description.

    Returns
    -------
    str
        The Markdown text.

    Examples
    --------
    >>> text = render_index(load_card_set(), DEFAULT_LINK_BASE, "Summary.")
    >>> text.splitlines()[0]
    '# TorchLens agent index'
    """
    lines = ["# TorchLens agent index", "", f"{summary}. Import as `import torchlens as tl`.", ""]
    lines += [f"Page paths below are relative to {link_base}", ""]
    lines += ["## Read first", ""] + [f"- {rule}" for rule in card_set.rules] + [""]
    lines += ["## Core calls", "", "```python", "import torchlens as tl"]
    lines += [c.canonical for c in card_set.cards if c.tier == "core" and c.canonical != NO_CALL]
    lines += ["```", "", "## Tasks", "", "| Task | Call | Avoid | Page |", "|---|---|---|---|"]
    for card in card_set.cards:
        lines.append(
            f"| {_cell(card.task)} | {_call_cell(card)} | {_cell(card.avoid)} | {card.page} |"
        )
    lines += [
        "",
        f"Capabilities by name: {CAPABILITIES_PATH.as_posix()}. "
        "Read-only tools over saved artifacts: `torchlens guide`.",
    ]
    return "\n".join(lines) + "\n"


def render_capabilities(card_set: CardSet, link_base: str) -> str:
    """Render the capabilities list page.

    Parameters
    ----------
    card_set : CardSet
        The loaded cards.
    link_base : str
        URL prefix for page links.

    Returns
    -------
    str
        The Markdown text.
    """
    by_slug = card_set.by_slug()
    lines = [
        "# Capabilities",
        "",
        "<!-- Generated by tools/docs/build_agent_index.py from "
        "docs/field-guide/task_cards.yaml; edit the cards, not this page. -->",
        "",
        "Each row is a capability name from Goodfire Silico's capabilities page "
        "(https://docs.goodfire.com/concepts/capabilities/), mapped to the TorchLens call "
        "for it. Rows without a call say where the work belongs instead.",
        "",
        "| Capability | TorchLens | Call |",
        "|---|---|---|",
    ]
    for row in card_set.capabilities:
        cards = [by_slug[slug] for slug in row.cards]
        tasks = "; ".join(f"[{_cell(c.task)}]({link_base}{c.page})" for c in cards)
        calls = "; ".join(_call_cell(c) for c in cards if c.canonical != NO_CALL)
        text = "; ".join(part for part in (tasks, _cell(row.note)) if part)
        lines.append(f"| {row.name} | {text} | {calls} |")
    return "\n".join(lines) + "\n"


def render_llms(card_set: CardSet, link_base: str, summary: str) -> str:
    """Render ``llms.txt``: one link line per card.

    Parameters
    ----------
    card_set : CardSet
        The loaded cards.
    link_base : str
        URL prefix for page links.
    summary : str
        The one-line project description.

    Returns
    -------
    str
        The llms.txt text.
    """
    lines = ["# TorchLens", "", f"> {summary}", "", "## Tasks", ""]
    lines += [f"- [{c.task}]({link_base}{c.page}): {c.one_liner}" for c in card_set.cards]
    lines += [
        "",
        "## Optional",
        "",
        f"- [Agent index]({link_base}{INDEX_PATH.as_posix()}): the compact task-to-call index",
        f"- [Capabilities]({link_base}{CAPABILITIES_PATH.as_posix()}): capability names mapped "
        "to calls",
    ]
    return "\n".join(lines) + "\n"


def _unique_pages(cards: Iterable[Card]) -> list[str]:
    pages: list[str] = []
    for card in cards:
        page = card.page.split("#")[0]
        if page not in pages:
            pages.append(page)
    return pages


def render_llms_full(card_set: CardSet, link_base: str, summary: str, repo: Path) -> str:
    """Render ``llms-full.txt``: the full Markdown of every linked page, front matter removed.

    Parameters
    ----------
    card_set : CardSet
        The loaded cards.
    link_base : str
        URL prefix for the source comment of each page.
    summary : str
        The one-line project description.
    repo : Path
        Repository root.

    Returns
    -------
    str
        The concatenated Markdown.
    """
    parts = [f"# TorchLens\n\n> {summary}\n"]
    for page in _unique_pages(card_set.cards):
        _, body = split_front_matter((repo / page).read_text("utf-8"))
        parts.append(f"<!-- source: {link_base}{page} -->\n\n{body.rstrip()}\n")
    return "\n".join(parts)


def render_all(card_set: CardSet, link_base: str, repo: Path = REPO) -> dict[Path, str]:
    """Render every output as ``{repository-relative path: text}``.

    Parameters
    ----------
    card_set : CardSet
        The loaded cards.
    link_base : str
        URL prefix for page links.
    repo : Path
        Repository root.

    Returns
    -------
    dict[Path, str]
        Every output file's text.

    Raises
    ------
    CardError
        When the agent index exceeds its 8 KB limit.
    """
    summary = _project_summary(repo)
    index = render_index(card_set, link_base, summary)
    size = len(index.encode("utf-8"))
    if size > INDEX_LIMIT_BYTES:
        raise CardError(f"AGENT_INDEX.md is {size} bytes; the limit is {INDEX_LIMIT_BYTES}")
    return {
        INDEX_PATH: index,
        ROUTER_PATH: index,
        CAPABILITIES_PATH: render_capabilities(card_set, link_base),
        LLMS_PATH: render_llms(card_set, link_base, summary),
    }


def stale_outputs(repo: Path, outputs: dict[Path, str]) -> list[Path]:
    """Return the outputs that are missing under ``repo`` or differ from ``outputs``.

    Parameters
    ----------
    repo : Path
        Repository root.
    outputs : dict[Path, str]
        Rendered outputs.

    Returns
    -------
    list[Path]
        The stale relative paths.
    """
    stale = []
    for rel, text in outputs.items():
        path = repo / rel
        if not path.exists() or path.read_text("utf-8") != text:
            stale.append(rel)
    return stale


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface.

    Parameters
    ----------
    argv : list[str] | None
        Arguments (``None`` = ``sys.argv[1:]``).

    Returns
    -------
    int
        0 when written or fresh, 1 when ``--check`` finds a stale output, 2 on a card error.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--link-base", default=DEFAULT_LINK_BASE)
    parser.add_argument("--check", action="store_true", help="fail on drift, write nothing")
    parser.add_argument(
        "--full",
        type=Path,
        default=None,
        help="also write llms-full.txt (every card page inlined) here; untracked, for site builds",
    )
    args = parser.parse_args(argv)
    try:
        card_set = load_card_set()
        outputs = render_all(card_set, args.link_base)
    except CardError as error:
        print(f"task card error: {error}", file=sys.stderr)
        return 2
    if args.full is not None and not args.check:
        summary = _project_summary(REPO)
        args.full.write_text(render_llms_full(card_set, args.link_base, summary, REPO), "utf-8")
    if args.check:
        stale = stale_outputs(REPO, outputs)
        for rel in stale:
            print(f"stale: {rel} (run python tools/docs/build_agent_index.py)")
        return 1 if stale else 0
    for rel, text in outputs.items():
        (REPO / rel).parent.mkdir(parents=True, exist_ok=True)
        (REPO / rel).write_text(text, "utf-8")
    print(f"wrote {len(outputs)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
