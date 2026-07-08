"""Validate committed menagerie JSONL catalog sources."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from menagerie.catalog import DEFERRED_JSONL, SOURCE_JSONL
from menagerie.schema import load_jsonl, validate_records


def validate_catalog_files(
    source_jsonl: Path = SOURCE_JSONL, deferred_jsonl: Path = DEFERRED_JSONL
) -> int:
    """Validate the committed JSONL catalog sources.

    Parameters
    ----------
    source_jsonl:
        Forward-required non-classics JSONL source.
    deferred_jsonl:
        Honestly deferred non-classics JSONL source.

    Returns
    -------
    int
        Number of validated records.
    """

    records = load_jsonl(source_jsonl)
    deferred_records = load_jsonl(deferred_jsonl) if deferred_jsonl.exists() else []
    validate_records([*records, *deferred_records])
    return len(records) + len(deferred_records)


def build_parser() -> argparse.ArgumentParser:
    """Build the catalog-validation CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, default=SOURCE_JSONL)
    parser.add_argument("--deferred-jsonl", type=Path, default=DEFERRED_JSONL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the catalog-validation CLI.

    Parameters
    ----------
    argv:
        Optional argument vector.

    Returns
    -------
    int
        Process exit status.
    """

    args = build_parser().parse_args(argv)
    count = validate_catalog_files(args.source_jsonl, args.deferred_jsonl)
    print(f"validated_records={count}")
    print(f"source_jsonl={args.source_jsonl}")
    print(f"deferred_jsonl={args.deferred_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
