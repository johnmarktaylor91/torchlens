"""Package-level command dispatcher for menagerie tools."""

from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch ``python -m menagerie`` subcommands.

    Parameters
    ----------
    argv:
        Optional argument vector.

    Returns
    -------
    int
        Process exit code.
    """

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "env":
        from menagerie.envs import main as env_main

        return env_main(args[1:])
    print("usage: python -m menagerie env <cmd>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
