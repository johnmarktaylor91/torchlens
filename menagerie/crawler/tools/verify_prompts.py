"""Verify crawler prompt files against PLAN section 18 frozen blocks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.identity import hash_bytes

_AUTHOR_HEADING = b"### 18.1 Claude crawler-author prompt"
_CHECKER_HEADING = b"### 18.2 Codex accuracy and fidelity checker prompt"
_TEXT_FENCE = b"```text\n"
_CLOSE_FENCE = b"\n```"


def build_parser() -> argparse.ArgumentParser:
    """Build the frozen-prompt verification parser.

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
    return parser


def _frozen_block(plan: bytes, heading: bytes) -> bytes:
    """Extract one exact newline-terminated frozen text block.

    Parameters
    ----------
    plan:
        Complete PLAN.md bytes.
    heading:
        Exact section heading preceding the frozen block.

    Returns
    -------
    bytes
        Bytes between the text fence and closing fence, including final newline.
    """

    heading_at = plan.index(heading)
    block_at = plan.index(_TEXT_FENCE, heading_at) + len(_TEXT_FENCE)
    block_end = plan.index(_CLOSE_FENCE, block_at) + 1
    return plan[block_at:block_end]


def verify_prompts(plan_path: Path, author_path: Path, checker_path: Path) -> dict[str, str]:
    """Verify both prompt files and return their SHA-256 digests.

    Parameters
    ----------
    plan_path, author_path, checker_path:
        Frozen plan and shipped prompt paths.

    Returns
    -------
    dict[str, str]
        Prompt filename to exact byte digest.

    Raises
    ------
    ValueError
        If either prompt differs from its frozen PLAN block.
    """

    plan = plan_path.read_bytes()
    paths = ((author_path, _AUTHOR_HEADING), (checker_path, _CHECKER_HEADING))
    digests: dict[str, str] = {}
    for path, heading in paths:
        actual = path.read_bytes()
        expected = _frozen_block(plan, heading)
        digests[path.name] = hash_bytes(actual)
        if actual != expected:
            raise ValueError(f"prompt drift: {path}")
    return digests


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Check both frozen prompts and print their byte hashes.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero when both prompts match, otherwise non-zero.
    """

    args = build_parser().parse_args(argv)
    try:
        digests = verify_prompts(args.plan, args.author_prompt, args.checker_prompt)
    except (OSError, ValueError) as exc:
        print(f"prompt verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"prompt_sha256": digests}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
