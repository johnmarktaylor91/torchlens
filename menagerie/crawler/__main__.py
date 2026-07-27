"""Module entry point for ``python -m menagerie.crawler``."""

from __future__ import annotations

import sys

from menagerie.crawler.cli import main


def _main() -> int:
    """Delegate process execution to the typed crawler CLI."""

    return main()


if __name__ == "__main__":
    sys.exit(_main())
