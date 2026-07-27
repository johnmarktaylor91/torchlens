"""Merge four completed crawler clones into validated derived views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

from menagerie.crawler.campaign_merge import (
    CampaignMergeError,
    CampaignSource,
    merge_campaigns,
)
from menagerie.crawler.partitioner import DEFAULT_CAMPAIGN_MANIFEST


def _campaign_argument(value: str) -> CampaignSource:
    """Parse ``CAMPAIGN_ID=CLONE_ROOT`` into a campaign source.

    Parameters
    ----------
    value:
        Command-line campaign binding.

    Returns
    -------
    CampaignSource
        Canonical records and runtime roots below the clone.
    """

    campaign_id, separator, raw_root = value.partition("=")
    if not separator or not campaign_id or not raw_root:
        raise argparse.ArgumentTypeError("campaign must be CAMPAIGN_ID=CLONE_ROOT")
    clone_root = Path(raw_root).expanduser().resolve()
    return CampaignSource(
        campaign_id=campaign_id,
        records_root=clone_root / "menagerie" / "crawler" / "records",
        runtime_root=clone_root / ".crawl-local",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the four-campaign merge command parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for manifest, clone roots, and derived output.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CAMPAIGN_MANIFEST)
    parser.add_argument(
        "--campaign",
        action="append",
        required=True,
        type=_campaign_argument,
        help="repeat exactly four times as CAMPAIGN_ID=CLONE_ROOT",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("menagerie/crawler/views/merged"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the strict read-only campaign merge.

    Parameters
    ----------
    argv:
        Optional arguments excluding the executable name.

    Returns
    -------
    int
        Zero after all proofs and view writes, otherwise one.
    """

    args = build_parser().parse_args(argv)
    try:
        result = merge_campaigns(args.manifest, args.campaign, args.output_root)
    except (CampaignMergeError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"campaign merge failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "terminal_count": len(result.current_records),
                "view_digests": result.view_digests,
                "risk_r5_alert": result.report["throughput"]["risk_r5"]["alert"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
