"""Regenerate the frozen root SURFACE table (C01 item 7).

One command: ``python tools/generate_surface.py``. Rerun after any
deliberate root-facade change (e.g. rebasing over a lane that added lazy
names) and review the diff; the lockstep test
(tests/test_arch_spine_surface.py) byte-checks the table against the live
walk in both directions.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from torchlens._surface import render_surface_tsv, surface_rows  # noqa: E402

TABLE = REPO / "tests" / "test_arch_spine_surface_table.tsv"


def main() -> None:
    rows = surface_rows()
    TABLE.write_text(
        "# test_arch_spine_surface_table.tsv -- FROZEN root SURFACE table (C01 item 7).\n"
        "# One row per reachable root public name; regenerated ONLY by\n"
        "# tools/generate_surface.py with the diff reviewed. legacy_top_level=1 rows are\n"
        "# grandfathered L6+ root names (contraction at next major); the ban on NEW L6+\n"
        "# root names is immediate (tests/test_arch_spine_surface.py).\n" + render_surface_tsv(rows)
    )
    print(f"wrote {TABLE} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
