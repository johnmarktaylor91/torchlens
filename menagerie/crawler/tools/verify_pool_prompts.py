"""Verify the whole crawler prompt surface, pool fragments included, against PLAN pins.

``verify_prompts`` pins the two top-level agent prompts. The dispatch-brief fragments under
``prompts/pool/`` shape every author session just as authoritatively, so they carry literal
PLAN digests under the same section and are checked by the same oracle. This module is the
superset command: it verifies the two top-level prompts *and* every shipped pool fragment,
and additionally proves the pinned inventory and the shipped inventory are the same set, so
neither adding an unpinned fragment nor deleting a pinned one can hide.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.tools.verify_prompts import verify_prompts

#: Digest rows are read from exactly one PLAN section, so an unrelated bullet elsewhere in
#: the document can never be mistaken for a pin and a pin can never be parked outside it.
PIN_SECTION_HEADING = "## 18. Frozen agent prompt identities"

_PIN_ROW = re.compile(r"^- `([^`\n]+)`: `[^`\n]*`", re.MULTILINE)
_NEXT_HEADING = re.compile(r"^## ", re.MULTILINE)


def build_parser() -> argparse.ArgumentParser:
    """Build the full prompt-surface verification parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with overridable paths for tests and audits.
    """

    crawler_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=crawler_root / "PLAN.md")
    parser.add_argument(
        "--author-prompt",
        type=Path,
        default=crawler_root / "prompts" / "claude_crawler_author_v2.txt",
    )
    parser.add_argument(
        "--checker-prompt",
        type=Path,
        default=crawler_root / "prompts" / "codex_accuracy_checker_v2.txt",
    )
    parser.add_argument(
        "--pool-root",
        type=Path,
        default=crawler_root / "prompts" / "pool",
    )
    return parser


def pinned_prompt_names(plan_path: Path) -> tuple[str, ...]:
    """Return every prompt name pinned in the frozen-identity section, in file order.

    Parameters
    ----------
    plan_path:
        Committed PLAN.md path.

    Returns
    -------
    tuple[str, ...]
        Pinned prompt file names.

    Raises
    ------
    ValueError
        If the pinned section is absent or pins the same name twice. A missing section is
        a removed drift oracle, which must fail rather than silently verify nothing.
    """

    plan = plan_path.read_text(encoding="utf-8")
    start = plan.find(PIN_SECTION_HEADING)
    if start < 0:
        raise ValueError(f"missing pinned prompt section: {PIN_SECTION_HEADING!r}")
    after = _NEXT_HEADING.search(plan, start + len(PIN_SECTION_HEADING))
    section = plan[start : after.start() if after else len(plan)]
    names = tuple(match.group(1) for match in _PIN_ROW.finditer(section))
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate pinned prompt digests: {', '.join(duplicates)}")
    return names


def pool_prompt_paths(pool_root: Path) -> tuple[Path, ...]:
    """Return every shipped pool fragment, sorted by name.

    Parameters
    ----------
    pool_root:
        ``prompts/pool`` directory rendered into every author dispatch brief.

    Returns
    -------
    tuple[Path, ...]
        Fragment paths.

    Raises
    ------
    ValueError
        If the directory is missing or ships no fragment. An empty pool cannot dispatch,
        so reporting it as verified would be a false pass.
    """

    if not pool_root.is_dir():
        raise ValueError(f"missing prompt pool directory: {pool_root}")
    paths = tuple(sorted((path for path in pool_root.glob("*.md") if path.is_file()), key=lambda p: p.name))
    if not paths:
        raise ValueError(f"prompt pool ships no fragment: {pool_root}")
    return paths


def verify_prompt_surface(
    plan_path: Path,
    author_path: Path,
    checker_path: Path,
    pool_root: Path,
) -> dict[str, str]:
    """Verify every top-level prompt and pool fragment against its literal PLAN digest.

    Parameters
    ----------
    plan_path, author_path, checker_path:
        Frozen plan and the two top-level prompt paths.
    pool_root:
        Directory of dispatch-brief fragments.

    Returns
    -------
    dict[str, str]
        Prompt filename to exact byte digest, covering the whole surface.

    Raises
    ------
    ValueError
        If a shipped prompt is unpinned, a pinned prompt is not shipped, or any prompt
        differs from its independently pinned PLAN digest.
    """

    fragments = pool_prompt_paths(pool_root)
    shipped = {path.name: path for path in (author_path, checker_path, *fragments)}
    if len(shipped) != 2 + len(fragments):
        raise ValueError("prompt file names collide across the pinned surface")
    pinned = set(pinned_prompt_names(plan_path))
    unpinned = sorted(set(shipped) - pinned)
    if unpinned:
        raise ValueError(f"shipped prompt is not pinned in PLAN: {', '.join(unpinned)}")
    unshipped = sorted(pinned - set(shipped))
    if unshipped:
        raise ValueError(f"pinned prompt is not shipped: {', '.join(unshipped)}")

    # verify_prompts is the single drift oracle: it hashes a file and compares it against
    # that file's own literal PLAN row. Pairing each fragment with the already-pinned
    # author prompt reuses that oracle verbatim instead of re-deriving a second comparison.
    digests = dict(verify_prompts(plan_path, author_path, checker_path))
    for fragment in fragments:
        digests.update(verify_prompts(plan_path, author_path, fragment))
    return digests


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Check the whole frozen prompt surface and print its byte hashes.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero when every prompt matches its pin, otherwise non-zero.
    """

    args = build_parser().parse_args(argv)
    try:
        digests = verify_prompt_surface(
            args.plan,
            args.author_prompt,
            args.checker_prompt,
            args.pool_root,
        )
    except (OSError, ValueError) as exc:
        print(f"prompt verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"prompt_sha256": digests}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
