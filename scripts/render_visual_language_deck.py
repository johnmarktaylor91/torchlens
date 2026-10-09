"""Build the TorchLens visual language deck: one slide per part of the visual language.

The third member of the ``render_collapse_reference.py`` / ``render_encoding_reference.py``
family. Every slide is a real TorchLens render of a tiny fixture model, with numbered
badges on the marks it teaches and a key beside the picture; the slide table, fixture
models, coverage map and composition live in ``scripts/visual_language/``.

Usage::

    python scripts/render_visual_language_deck.py discover   # universes and unclassified items
    python scripts/render_visual_language_deck.py check      # coverage map + slide table, no render
    python scripts/render_visual_language_deck.py render --out DIR   # panels only (SVG, DOT, JSON)
    python scripts/render_visual_language_deck.py deck --out DIR     # full deck and receipts

``deck`` exits non-zero when any slide fails its selectors, legibility check or render;
the failing deck is still written so the failures can be read (``slides.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def check_table() -> list[str]:
    """Static checks of the slide table: fixtures exist, selectors parse, ids are unique."""

    from scripts.visual_language.dot import parse_selector
    from scripts.visual_language.slides import ALPHABET, SLIDES, fill, slide_ids

    problems: list[str] = []
    ids = slide_ids()
    if len(set(ids)) != len(ids):
        problems.append("duplicate slide ids")
    for slide in SLIDES:
        panels = {panel.name for panel in slide.panels}
        selectors = [(k.panel, k.select) for k in slide.keys if k.select]
        selectors += [(c.panel, c.select) for c in slide.cells if c.select]
        for panel_name, selector in selectors:
            if panel_name not in panels:
                problems.append(f"{slide.id}: key names missing panel {panel_name}")
            try:
                parse_selector(fill(selector))
            except (ValueError, KeyError) as exc:
                problems.append(f"{slide.id}: {exc}")
    for sheet, entries in ALPHABET.items():
        for _meaning, source, _panel, selector in entries:
            if source not in ids:
                problems.append(f"{sheet}: unknown slide {source}")
            parse_selector(fill(selector))
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand; return the process exit code."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("discover")
    sub.add_parser("check")
    for name in ("render", "deck"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--only", nargs="*", default=None, help="slide ids to build")
    args = parser.parse_args(argv)

    if args.command == "discover":
        from scripts.visual_language import coverage

        print(json.dumps(coverage.discover(), indent=1, default=str))
        return 0
    if args.command == "check":
        from scripts.visual_language import coverage

        problems = check_table() + coverage.check_all()
        for line in problems:
            print(line)
        print(f"{len(problems)} problem(s)")
        return 1 if problems else 0

    from scripts.visual_language import deck

    only = set(args.only) if args.only else None
    if args.command == "render":
        records = deck.render_all(args.out / "raw", only)
        errors = {k: v["error"] for k, v in records.items() if "error" in v}
        print(json.dumps({"panels": len(records), "errors": errors}, indent=1))
        return 1 if errors else 0
    return 1 if deck.build(args.out, only) else 0


if __name__ == "__main__":
    raise SystemExit(main())
