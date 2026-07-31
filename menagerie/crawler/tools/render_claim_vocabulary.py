"""Render the gated-claim vocabulary into the author prompt from the enforcing code.

The author prompt must state the closed set of claim-category strings a ``supports``
entry may carry, because the coverage gate matches those strings by exact equality and
they exist nowhere the author can see. Hand-copying that list is what produced the wall
this tool removes: the list drifts, the author guesses, and every proposal is refused
with ``ungrounded claim categories``.

So the region is GENERATED from :data:`menagerie.crawler.proposal.DEFAULT_GATED_CLAIMS`
and re-derived by the test suite on every run. Changing the gated set without
regenerating the prompt is a hard test failure, not a silent divergence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.constants import AUTHOR_PROMPT_NAME
from menagerie.crawler.proposal import (
    CLAIM_VOCABULARY_BEGIN,
    CLAIM_VOCABULARY_END,
    gated_claim_vocabulary_block,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the vocabulary-rendering parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with an overridable prompt path for tests and audits.
    """

    crawler_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        type=Path,
        default=crawler_root / "prompts" / f"{AUTHOR_PROMPT_NAME}.txt",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the generated region in place instead of only checking it.",
    )
    return parser


def render(prompt_text: str) -> str:
    """Return the prompt with its generated vocabulary region freshly derived.

    Parameters
    ----------
    prompt_text:
        Complete current prompt text.

    Returns
    -------
    str
        Prompt text whose generated region matches the enforcing code.

    Raises
    ------
    ValueError
        If the delimited region is absent or malformed.
    """

    begin = prompt_text.find(CLAIM_VOCABULARY_BEGIN)
    end = prompt_text.find(CLAIM_VOCABULARY_END)
    if begin < 0 or end < 0 or end < begin:
        raise ValueError("author prompt has no well-formed generated claim-vocabulary region")
    tail = end + len(CLAIM_VOCABULARY_END)
    return prompt_text[:begin] + gated_claim_vocabulary_block() + prompt_text[tail:]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Check or rewrite the prompt's generated claim-vocabulary region.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero when the region already matches, or when it was rewritten.
    """

    args = build_parser().parse_args(argv)
    try:
        current = args.prompt.read_text(encoding="utf-8")
        rendered = render(current)
    except (OSError, ValueError) as exc:
        print(f"claim-vocabulary rendering failed: {exc}", file=sys.stderr)
        return 1
    if rendered == current:
        print("claim vocabulary is current")
        return 0
    if not args.write:
        print(
            "author prompt claim vocabulary is stale; rerun with --write",
            file=sys.stderr,
        )
        return 1
    args.prompt.write_text(rendered, encoding="utf-8")
    print(f"rewrote claim vocabulary in {args.prompt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
