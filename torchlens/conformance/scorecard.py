"""``python -m torchlens.conformance.scorecard``: compute, write, check or grade a scorecard.

Usage::

    python -m torchlens.conformance.scorecard --backend tf              # print the record
    python -m torchlens.conformance.scorecard --backend tf --write      # write the shipped file
    python -m torchlens.conformance.scorecard --backend tf --check      # compare with the shipped file
    python -m torchlens.conformance.scorecard --backend tf --graduation # print the six items

``--write`` and ``--check`` belong to the backend's pinned CI leg: the
shipped record (``torchlens/backends/<backend>/scorecard.json``) is a tested
fact, never hand-edited, and ``--check`` exits 1 when the recomputed record
differs from it. ``--graduation`` reads the shipped record only.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._scorecard import PreviewScorecard

__tl_layer__ = "L9"


def _parser() -> argparse.ArgumentParser:
    """The command-line parser."""

    parser = argparse.ArgumentParser(
        prog="python -m torchlens.conformance.scorecard",
        description="Compute, write, check or grade one backend's conformance scorecard.",
    )
    parser.add_argument("--backend", required=True, help="registered backend name")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write the shipped record")
    mode.add_argument("--check", action="store_true", help="compare against the shipped record")
    mode.add_argument(
        "--graduation",
        action="store_true",
        help="print the graduation report from the shipped record",
    )
    parser.add_argument(
        "--workdir", type=Path, default=None, help="scratch directory for c2 artifacts"
    )
    parser.add_argument(
        "--repo-root", type=Path, default=None, help="repository root for the file-based items"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command.

    Parameters
    ----------
    argv:
        Arguments without the program name (``sys.argv[1:]`` when omitted).

    Returns
    -------
    int
        ``0`` on success, ``1`` when ``--check`` finds a difference, ``2`` when
        a shipped record is required and missing.
    """

    from ._graduation import _graduation_report
    from ._scorecard import _preview_scorecard, shipped_scorecard, shipped_scorecard_path

    args = _parser().parse_args(argv)
    if args.graduation:
        if shipped_scorecard(args.backend) is None:
            print(
                f"no shipped scorecard for {args.backend}: {shipped_scorecard_path(args.backend)}"
            )
            return 2
        print(_graduation_report(args.backend, repo_root=args.repo_root).render())
        return 0
    card = _preview_scorecard(args.backend, workdir=args.workdir, repo_root=args.repo_root)
    if args.write:
        path = shipped_scorecard_path(args.backend)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(card.to_json() + "\n", encoding="utf-8")
        print(f"wrote {path} ({card.counts['passed']}/{card.counts['denominator']} cells passed)")
        return 0
    if args.check:
        return _check_shipped(args.backend, card)
    print(card.to_json())
    return 0


def _check_shipped(backend: str, card: PreviewScorecard) -> int:
    """Compare the shipped record with ``card``: ``0`` equal, ``1`` differs, ``2`` none ships."""

    from ._scorecard import scorecards_differ, shipped_scorecard, shipped_scorecard_path

    shipped = shipped_scorecard(backend)
    if shipped is None:
        print(f"no shipped scorecard for {backend}: {shipped_scorecard_path(backend)}")
        return 2
    differences = scorecards_differ(shipped, card)
    if differences:
        print(f"shipped scorecard for {backend} differs from the computed one:")
        print("\n".join(f"  {line}" for line in differences))
        return 1
    print(f"shipped scorecard for {backend} matches ({card.counts['denominator']} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
