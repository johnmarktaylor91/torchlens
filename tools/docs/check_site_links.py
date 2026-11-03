"""Check internal links and anchors in a built static site, offline.

Every ``href`` and ``src`` in every HTML file under the site directory that
points inside the site must resolve to a file (a directory resolves to its
``index.html``), and a ``#fragment`` must name an ``id`` (or ``name``) on the
target page. External links (with a scheme or ``//``) are not fetched.

Usage::

    python tools/docs/check_site_links.py site/
"""

from __future__ import annotations

import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

#: Generated files that link by design to paths the site does not contain.
SKIP_SOURCES: tuple[str, ...] = ("404.html",)


class _PageScan(HTMLParser):
    """Collect the link targets and the anchor ids of one page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is None:
                continue
            if name in ("id", "name"):
                self.ids.add(value)
            elif name in ("href", "src"):
                self.links.append(value)


def _scan(page: Path, cache: dict[Path, _PageScan]) -> _PageScan:
    if page not in cache:
        scan = _PageScan()
        scan.feed(page.read_text("utf-8", errors="replace"))
        cache[page] = scan
    return cache[page]


def _target_file(site: Path, page: Path, path: str) -> Path:
    base = site if path.startswith("/") else page.parent
    target = (base / unquote(path.lstrip("/"))).resolve()
    return target / "index.html" if target.is_dir() else target


def check_site(site: Path, base_path: str = "/") -> list[str]:
    """Return one message per broken internal link or anchor.

    Parameters
    ----------
    site : Path
        The built site directory.
    base_path : str
        URL path the site is served under (``site_url``'s path); absolute links
        under it are checked against the site directory.

    Returns
    -------
    list[str]
        Problems as ``page: link (reason)``; empty when every link resolves.
    """
    site = site.resolve()
    cache: dict[Path, _PageScan] = {}
    problems = []
    for page in sorted(site.rglob("*.html")):
        if page.name in SKIP_SOURCES:
            continue
        for link in _scan(page, cache).links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc or link.startswith(("mailto:", "javascript:")):
                continue
            path = parts.path
            if path.startswith("/"):
                if not path.startswith(base_path):
                    problems.append(f"{page.relative_to(site)}: {link} (outside {base_path})")
                    continue
                path = "/" + path[len(base_path) :]
            target = _target_file(site, page, path) if path else page
            if not target.exists():
                problems.append(f"{page.relative_to(site)}: {link} (no such file)")
                continue
            fragment = unquote(parts.fragment)
            if fragment and target.suffix == ".html" and fragment not in _scan(target, cache).ids:
                problems.append(f"{page.relative_to(site)}: {link} (no anchor #{fragment})")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("site", type=Path)
    parser.add_argument("--base-path", default="/", help="URL path the site is served under")
    args = parser.parse_args(argv)
    problems = check_site(args.site, args.base_path)
    for problem in problems:
        print(problem, file=sys.stderr)
    pages = sum(1 for _ in args.site.rglob("*.html"))
    print(f"checked {pages} pages: {len(problems)} broken links")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
