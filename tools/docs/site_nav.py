"""Generate the docs site navigation, exclusions and redirects from the site map.

``docs/_site_map.yaml`` lists every page under ``docs/`` once, with its section
of the site map, whether it is published now, and any old paths it moved from.
This script renders three marked blocks of ``mkdocs.yml`` from it:

* ``nav``: the published pages, in section order, grouped where a page names a
  ``group``;
* ``exclude_docs``: every unpublished page, so the strict build never sees it;
* ``redirect_maps``: old path to new path for every published page that moved.

Section titles live only in the site map, so renaming a section (for example
the pencilled-in "Field guide") is one edit there plus a rerun of this script.

Usage::

    python tools/docs/site_nav.py           # rewrite the blocks in mkdocs.yml
    python tools/docs/site_nav.py --check   # exit 1 if mkdocs.yml or the map is stale
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
SITE_MAP = DOCS / "_site_map.yaml"
MKDOCS = REPO / "mkdocs.yml"

#: Files under docs/ that are site inputs but not pages, never listed in the map.
NON_PAGE_PATTERNS: tuple[str, ...] = ("_site_map.yaml",)


@dataclass(frozen=True)
class Page:
    """One row of the site map."""

    path: str
    section: str
    publish: bool
    title: str | None = None
    group: str | None = None
    moved_from: tuple[str, ...] = ()
    move_to: str | None = None


@dataclass(frozen=True)
class SiteMap:
    """The parsed site map: ordered sections and every page."""

    sections: dict[str, str]
    internal_sections: frozenset[str]
    pages: tuple[Page, ...] = field(default=())


def load_site_map(path: Path = SITE_MAP) -> SiteMap:
    """Parse and validate the site map file.

    Parameters
    ----------
    path : Path
        The YAML site map.

    Returns
    -------
    SiteMap
        The validated map.

    Raises
    ------
    ValueError
        On an unknown section, a duplicate page or a published internal page.
    """
    raw = yaml.safe_load(path.read_text("utf-8"))
    sections = {row["id"]: row["title"] for row in raw["sections"]}
    internal = frozenset(row["id"] for row in raw["sections"] if row.get("internal"))
    pages = []
    seen: set[str] = set()
    for row in raw["pages"]:
        page = Page(
            path=row["path"],
            section=row["section"],
            publish=bool(row.get("publish", False)),
            title=row.get("title"),
            group=row.get("group"),
            moved_from=tuple(row.get("moved_from", ())),
            move_to=row.get("move_to"),
        )
        if page.section not in sections:
            raise ValueError(f"{page.path}: unknown section {page.section!r}")
        if page.path in seen:
            raise ValueError(f"{page.path}: listed twice")
        if page.publish and page.section in internal:
            raise ValueError(f"{page.path}: section {page.section!r} is never published")
        seen.add(page.path)
        pages.append(page)
    return SiteMap(sections=sections, internal_sections=internal, pages=tuple(pages))


def coverage_problems(site_map: SiteMap, docs: Path = DOCS) -> list[str]:
    """Return pages on disk missing from the map and map rows with no page on disk."""
    on_disk = {p.relative_to(docs).as_posix() for p in docs.rglob("*.md")}
    listed = {page.path for page in site_map.pages}
    problems = [f"not in the site map: {path}" for path in sorted(on_disk - listed)]
    problems += [f"in the site map but missing: {path}" for path in sorted(listed - on_disk)]
    return problems


def _nav_entry(page: Page) -> str:
    return f"{page.title}: {page.path}" if page.title else page.path


def render_nav(site_map: SiteMap) -> list[str]:
    """Render the ``nav:`` block body as YAML lines."""
    lines: list[str] = []
    for section_id, title in site_map.sections.items():
        published = [p for p in site_map.pages if p.section == section_id and p.publish]
        if not published:
            continue
        if section_id == "home":
            lines += [f"  - {_nav_entry(p)}" for p in published]
            continue
        lines.append(f"  - {title}:")
        groups: dict[str | None, list[Page]] = {}
        for page in published:
            groups.setdefault(page.group, []).append(page)
        lines += [f"      - {_nav_entry(p)}" for p in groups.pop(None, [])]
        for group, members in groups.items():
            lines.append(f"      - {group}:")
            lines += [f"          - {_nav_entry(p)}" for p in members]
    return lines


def render_excludes(site_map: SiteMap) -> list[str]:
    """Render the ``exclude_docs`` patterns (one unpublished page per line)."""
    lines = [f"  /{pattern}" for pattern in NON_PAGE_PATTERNS]
    lines += [f"  /{p.path}" for p in site_map.pages if not p.publish]
    return lines


def render_redirects(site_map: SiteMap) -> list[str]:
    """Render ``redirect_maps`` entries for published pages that moved."""
    lines = []
    for page in site_map.pages:
        if page.publish:
            lines += [f"        {old}: {page.path}" for old in page.moved_from]
    return lines or ["        {}"]


def _replace_block(text: str, name: str, body: list[str]) -> str:
    pattern = re.compile(
        rf"(^[ \t]*# <generated:{name}>\n).*?(^[ \t]*# </generated:{name}>$)",
        re.MULTILINE | re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError(f"mkdocs.yml has no generated:{name} markers")
    return pattern.sub(lambda m: m.group(1) + "\n".join(body) + "\n" + m.group(2), text)


def render_mkdocs(text: str, site_map: SiteMap) -> str:
    """Return ``mkdocs.yml`` text with every generated block refreshed."""
    text = _replace_block(text, "nav", render_nav(site_map))
    text = _replace_block(text, "exclude_docs", render_excludes(site_map))
    return _replace_block(text, "redirects", render_redirects(site_map))


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="fail on drift, write nothing")
    args = parser.parse_args(argv)

    site_map = load_site_map()
    problems = coverage_problems(site_map)
    current = MKDOCS.read_text("utf-8")
    fresh = render_mkdocs(current, site_map)
    if args.check:
        if fresh != current:
            problems.append("mkdocs.yml is stale: run python tools/docs/site_nav.py")
    elif fresh != current:
        MKDOCS.write_text(fresh, "utf-8")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
