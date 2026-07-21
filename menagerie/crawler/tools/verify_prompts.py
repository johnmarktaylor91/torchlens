"""Verify crawler prompt files against PLAN-pinned literal SHA-256 values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.identity import hash_bytes, is_sha256

_HASH_ROW_PREFIX = "- `{prompt_name}`: `"


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


def _pinned_digest(plan: bytes, prompt_name: str) -> str:
    """Extract and validate one literal prompt digest from the committed plan.

    Parameters
    ----------
    plan:
        Complete PLAN.md bytes.
    prompt_name:
        Exact prompt filename named by the digest row.

    Returns
    -------
    str
        Canonical prefixed SHA-256 digest.

    Raises
    ------
    ValueError
        If the digest row is absent, malformed, or not ASCII.
    """

    marker = _HASH_ROW_PREFIX.format(prompt_name=prompt_name).encode("ascii")
    try:
        digest_at = plan.index(marker) + len(marker)
        digest_end = plan.index(b"`", digest_at)
        digest = plan[digest_at:digest_end].decode("ascii")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"missing or malformed pinned prompt digest: {prompt_name}") from exc
    if not is_sha256(digest):
        raise ValueError(f"noncanonical pinned prompt digest: {prompt_name}")
    return digest


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
        If either prompt differs from its independently pinned PLAN digest.
    """

    plan = plan_path.read_bytes()
    paths = (author_path, checker_path)
    digests: dict[str, str] = {}
    for path in paths:
        actual = hash_bytes(path.read_bytes())
        expected = _pinned_digest(plan, path.name)
        digests[path.name] = actual
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
