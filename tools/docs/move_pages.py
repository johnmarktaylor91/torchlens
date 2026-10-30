"""Move documentation pages and rewrite every reference to them.

A move runs in two separable steps so the renames can land as their own commit
(a pure-rename commit rebases cleanly under later codemods):

``--rename-only``
    ``git mv`` each source page to its destination. Nothing else changes.
``--rewrite-only``
    Rewrite references in tracked text files: repository-relative paths
    (``docs/old/page.md``), docs-relative literals (``"old/page.md"``) and
    relative Markdown links from any page, including links inside the moved
    pages whose depth changed.

With neither flag both steps run. The script is re-runnable: a page already at
its destination is not moved again, and a rewrite that finds nothing to change
leaves the tree untouched. ``--print-redirects`` prints the redirect map the
docs build needs for the old URLs.

Examples
--------
::

    python tools/docs/move_pages.py docs/recipes docs/field-guide --rename-only
    python tools/docs/move_pages.py docs/recipes docs/field-guide --rewrite-only
    python tools/docs/move_pages.py docs/reference/trace_core_design.md \\
        docs/internal/reference/trace_core_design.md --dry-run
"""

from __future__ import annotations

import argparse
import os
import posixpath
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = "docs"

#: Files whose old paths are history, not live references.
SKIP_FILES: frozenset[str] = frozenset({"CHANGELOG.md"})

_MD_LINK = re.compile(r"(\]\()([^)\s#]+)(#[^)\s]*)?(\))")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
    ).stdout


def plan_moves(src: str, dst: str) -> dict[str, str]:
    """Return ``{old repo path: new repo path}`` for one source and destination.

    Parameters
    ----------
    src : str
        Repository-relative page or directory to move.
    dst : str
        Repository-relative destination page or directory.

    Returns
    -------
    dict[str, str]
        Every Markdown page under ``src`` (tracked, or already moved to ``dst``)
        mapped to its destination.
    """
    src, dst = src.rstrip("/"), dst.rstrip("/")
    if src.endswith(".md"):
        return {src: dst}
    tracked = _git("ls-files", "--", src, dst).splitlines()
    moves: dict[str, str] = {}
    for path in tracked:
        if not path.endswith(".md"):
            continue
        rel = posixpath.relpath(path, src if path.startswith(src + "/") else dst)
        moves[posixpath.join(src, rel)] = posixpath.join(dst, rel)
    return moves


def rename(moves: dict[str, str], dry_run: bool) -> list[str]:
    """``git mv`` each page not yet at its destination; return the moved pages."""
    done = []
    for old, new in sorted(moves.items()):
        if not (REPO / old).exists() or (REPO / new).exists():
            continue
        done.append(old)
        if not dry_run:
            (REPO / new).parent.mkdir(parents=True, exist_ok=True)
            _git("mv", old, new)
    return done


def _docs_rel(path: str) -> str:
    return posixpath.relpath(path, DOCS)


def _rewrite_links(text: str, here_old: str, here_new: str, moves: dict[str, str]) -> str:
    """Re-target relative Markdown links in one page that moved or links to a moved page."""

    def fix(match: re.Match[str]) -> str:
        target = match.group(2)
        if "://" in target or target.startswith(("/", "mailto:")):
            return match.group(0)
        from_here = posixpath.normpath(posixpath.join(posixpath.dirname(here_new), target))
        if from_here not in moves and (REPO / from_here).exists():
            return match.group(0)  # already correct: keeps a second run a no-op
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(here_old), target))
        new_target_path = moves.get(resolved)
        if new_target_path is None:
            if here_old == here_new:
                return match.group(0)
            new_target_path = resolved
        new_target = posixpath.relpath(new_target_path, posixpath.dirname(here_new))
        return f"{match.group(1)}{new_target}{match.group(3) or ''}{match.group(4)}"

    return _MD_LINK.sub(fix, text)


def rewrite_text(text: str, path: str, moves: dict[str, str]) -> str:
    """Return ``text`` (the content of repo file ``path``) with references re-targeted.

    Parameters
    ----------
    text : str
        Current file content.
    path : str
        Repository-relative path of the file as it is now (after any rename).
    moves : dict[str, str]
        ``{old repo path: new repo path}``.

    Returns
    -------
    str
        The rewritten content.
    """
    inverse = {new: old for old, new in moves.items()}
    here_old = inverse.get(path, path)
    for old, new in moves.items():
        text = re.sub(rf"(?<![\w./-]){re.escape(old)}\b", new, text)
        old_rel, new_rel = _docs_rel(old), _docs_rel(new)
        text = re.sub(rf"(?<=[\"'`]){re.escape(old_rel)}(?=[\"'`#])", new_rel, text)
    if path.endswith(".md") and path.startswith(DOCS + "/"):
        text = _rewrite_links(text, here_old, path, moves)
    return text


def rewrite(moves: dict[str, str], dry_run: bool) -> list[str]:
    """Rewrite references across tracked text files; return the changed files."""
    changed = []
    for path in _git("ls-files").splitlines():
        if path in SKIP_FILES or path.startswith("tests/producer_parity/"):
            continue
        file = REPO / path
        try:
            text = file.read_text("utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        new_text = rewrite_text(text, path, moves)
        if new_text != text:
            changed.append(path)
            if not dry_run:
                file.write_text(new_text, "utf-8")
    return changed


def redirect_map(moves: dict[str, str]) -> dict[str, str]:
    """Return the docs-relative ``{old page: new page}`` map for the redirects plugin."""
    return {_docs_rel(old): _docs_rel(new) for old, new in sorted(moves.items())}


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    step = parser.add_mutually_exclusive_group()
    step.add_argument("--rename-only", action="store_true")
    step.add_argument("--rewrite-only", action="store_true")
    step.add_argument("--print-redirects", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    moves = plan_moves(args.src, args.dst)
    if not moves:
        print(f"no Markdown pages under {args.src} or {args.dst}", file=sys.stderr)
        return 1
    if args.print_redirects:
        for old, new in redirect_map(moves).items():
            print(f"{old}: {new}")
        return 0
    prefix = "would " if args.dry_run else ""
    if not args.rewrite_only:
        moved = rename(moves, args.dry_run)
        print(f"{prefix}move {len(moved)} of {len(moves)} pages")
    if not args.rename_only:
        changed = rewrite(moves, args.dry_run)
        print(f"{prefix}rewrite {len(changed)} files")
        for path in changed:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    os.chdir(REPO)
    raise SystemExit(main())
