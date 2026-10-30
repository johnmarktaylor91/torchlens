"""Generate ``llms.txt``, ``llms-full.txt`` and ``AGENT_INDEX.md`` from task cards.

A task card is the YAML front matter of a Field guide page (``docs/field-guide``).
This script reads the cards directly and needs no docs build tool, so it runs
before the site build and its outputs never depend on how the site is built.

Card fields: ``task``, ``one_liner``, ``canonical`` (one call), ``avoid``,
``tier`` (``core``, ``kit`` or ``advanced``), ``doors`` (from ``mechinterp``,
``neuroai``, ``vision``, ``agents``), ``related`` (page slugs) and optional
``tags``. Pages without front matter, or whose front matter has no ``task``,
carry no card and are skipped.

Outputs:

``AGENT_INDEX.md``
    At most 8 KB: the core calls, a task-to-call table with each card's
    ``avoid`` line, and links to the version-matched pages.
``llms.txt``
    The llms.txt index: one link line per card.
``llms-full.txt``
    The full Markdown of every carded page, front matter removed.

Usage::

    python tools/docs/build_agent_index.py                 # write to tools/docs/generated
    python tools/docs/build_agent_index.py --check         # exit 1 if those files are stale
    python tools/docs/build_agent_index.py --cards DIR --out DIR --version 2.x
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CARDS = REPO / "docs" / "field-guide"
DEFAULT_OUT = REPO / "tools" / "docs" / "generated"
SITE_MAP = REPO / "docs" / "_site_map.yaml"
DEFAULT_SITE_URL = "https://johnmarktaylor91.github.io/torchlens/"
DEFAULT_VERSION = "2.x"
INDEX_LIMIT_BYTES = 8 * 1024

TIERS: tuple[str, ...] = ("core", "kit", "advanced")
DOORS: frozenset[str] = frozenset({"mechinterp", "neuroai", "vision", "agents"})
REQUIRED: tuple[str, ...] = ("task", "one_liner", "canonical", "avoid", "tier", "doors", "related")
OPTIONAL: tuple[str, ...] = ("tags",)


class CardError(ValueError):
    """A task card is malformed."""


@dataclass(frozen=True)
class Card:
    """One task card and the page it came from."""

    slug: str
    task: str
    one_liner: str
    canonical: str
    avoid: str
    tier: str
    doors: tuple[str, ...]
    related: tuple[str, ...]
    tags: tuple[str, ...]
    body: str


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


def _string_list(slug: str, key: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise CardError(f"{slug}: {key} must be a list of strings")
    return tuple(value)


def card_from_meta(slug: str, meta: dict[str, Any], body: str) -> Card:
    """Build a card from page front matter, checking the field contract.

    Parameters
    ----------
    slug : str
        Page slug (file name without ``.md``).
    meta : dict[str, Any]
        The page's front matter.
    body : str
        The page Markdown after the front matter.

    Returns
    -------
    Card
        The checked card.

    Raises
    ------
    CardError
        On a missing, unknown or ill-typed field.
    """
    missing = [key for key in REQUIRED if key not in meta]
    unknown = sorted(set(meta) - set(REQUIRED) - set(OPTIONAL))
    if missing or unknown:
        raise CardError(f"{slug}: missing {missing}, unknown {unknown}")
    for key in ("task", "one_liner", "canonical", "avoid"):
        if not isinstance(meta[key], str) or not meta[key].strip():
            raise CardError(f"{slug}: {key} must be a non-empty string")
    if meta["tier"] not in TIERS:
        raise CardError(f"{slug}: tier must be one of {TIERS}")
    doors = _string_list(slug, "doors", meta["doors"])
    if not set(doors) <= DOORS:
        raise CardError(f"{slug}: unknown doors {sorted(set(doors) - DOORS)}")
    return Card(
        slug=slug,
        task=meta["task"].strip(),
        one_liner=meta["one_liner"].strip(),
        canonical=meta["canonical"].strip(),
        avoid=meta["avoid"].strip(),
        tier=meta["tier"],
        doors=doors,
        related=_string_list(slug, "related", meta["related"]),
        tags=_string_list(slug, "tags", meta.get("tags", [])),
        body=body,
    )


def load_cards(cards_dir: Path) -> list[Card]:
    """Read every carded page in ``cards_dir``, ordered by tier, then slug.

    Raises
    ------
    CardError
        On a malformed card, or a ``related`` slug that names no carded page.
    """
    cards = []
    for page in sorted(cards_dir.glob("*.md")):
        meta, body = split_front_matter(page.read_text("utf-8"))
        if meta is None or "task" not in meta:
            continue
        cards.append(card_from_meta(page.stem, meta, body))
    slugs = {card.slug for card in cards}
    for card in cards:
        stray = sorted(set(card.related) - slugs)
        if stray:
            raise CardError(f"{card.slug}: related names no carded page: {stray}")
    return sorted(cards, key=lambda c: (TIERS.index(c.tier), c.slug))


def _project_summary() -> str:
    text = (REPO / "pyproject.toml").read_text("utf-8")
    match = re.search(r'^description\s*=\s*"(.*)"\s*$', text, re.MULTILINE)
    if match is None:
        raise CardError("pyproject.toml has no one-line project description")
    return match.group(1).strip()


def _section_title(section_id: str, fallback: str) -> str:
    if not SITE_MAP.exists():
        return fallback
    for row in yaml.safe_load(SITE_MAP.read_text("utf-8"))["sections"]:
        if row["id"] == section_id:
            return str(row["title"])
    return fallback


def _page_url(site_url: str, version: str, slug: str) -> str:
    return f"{site_url.rstrip('/')}/{version}/field-guide/{slug}/"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_index(cards: list[Card], site_url: str, version: str) -> str:
    """Render ``AGENT_INDEX.md``."""
    guide = _section_title("field-guide", "Field guide")
    lines = [f"# TorchLens agent index ({version})", "", _project_summary(), ""]
    lines += ["## Core calls", "", "```python"]
    lines += [card.canonical for card in cards if card.tier == "core"] or ["# (no core cards yet)"]
    lines += ["```", "", "## Tasks", ""]
    lines += ["| Task | Call | Avoid | Tier | Page |", "|---|---|---|---|---|"]
    for card in cards:
        url = _page_url(site_url, version, card.slug)
        lines.append(
            f"| {_cell(card.task)} | `{_cell(card.canonical)}` | {_cell(card.avoid)} "
            f"| {card.tier} | [{guide}: {card.slug}]({url}) |"
        )
    return "\n".join(lines) + "\n"


def render_llms(cards: list[Card], site_url: str, version: str) -> str:
    """Render ``llms.txt``."""
    guide = _section_title("field-guide", "Field guide")
    lines = ["# TorchLens", "", f"> {_project_summary()}", "", f"## {guide}", ""]
    for card in cards:
        url = _page_url(site_url, version, card.slug)
        lines.append(f"- [{card.task}]({url}): {card.one_liner}")
    return "\n".join(lines) + "\n"


def render_llms_full(cards: list[Card], site_url: str, version: str) -> str:
    """Render ``llms-full.txt``."""
    parts = [f"# TorchLens\n\n> {_project_summary()}\n"]
    for card in cards:
        url = _page_url(site_url, version, card.slug)
        parts.append(f"<!-- source: {url} -->\n\n{card.body.rstrip()}\n")
    return "\n".join(parts)


def render_all(cards: list[Card], site_url: str, version: str) -> dict[str, str]:
    """Render every output as ``{file name: text}``.

    Raises
    ------
    CardError
        When the agent index exceeds its 8 KB limit.
    """
    outputs = {
        "AGENT_INDEX.md": render_index(cards, site_url, version),
        "llms.txt": render_llms(cards, site_url, version),
        "llms-full.txt": render_llms_full(cards, site_url, version),
    }
    size = len(outputs["AGENT_INDEX.md"].encode("utf-8"))
    if size > INDEX_LIMIT_BYTES:
        raise CardError(f"AGENT_INDEX.md is {size} bytes; the limit is {INDEX_LIMIT_BYTES}")
    return outputs


def stale_outputs(out_dir: Path, outputs: dict[str, str]) -> list[str]:
    """Return the output files in ``out_dir`` that are missing or differ from ``outputs``."""
    stale = []
    for name, text in outputs.items():
        path = out_dir / name
        if not path.exists() or path.read_text("utf-8") != text:
            stale.append(name)
    return stale


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--site-url", default=DEFAULT_SITE_URL)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--check", action="store_true", help="fail on drift, write nothing")
    args = parser.parse_args(argv)
    try:
        outputs = render_all(load_cards(args.cards), args.site_url, args.version)
    except CardError as error:
        print(f"task card error: {error}", file=sys.stderr)
        return 2
    if args.check:
        stale = stale_outputs(args.out, outputs)
        for name in stale:
            print(f"stale: {args.out / name} (run python tools/docs/build_agent_index.py)")
        return 1 if stale else 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (args.out / name).write_text(text, "utf-8")
    print(f"wrote {len(outputs)} files to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
