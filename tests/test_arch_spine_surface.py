"""Root SURFACE lockstep: walk == rows, budget frozen, L6+ ban (items 7 + 9).

The frozen table (test_arch_spine_surface_table.tsv) is regenerated only by
``tools/generate_surface.py`` with the diff reviewed; this suite re-derives
the live rows and demands byte equality in both directions, pins the root
budget, bans NEW L6+ root names (legacy rows are grandfathered; contraction
waits for the next major), and resolves every lazy name in a FRESH
interpreter (warm sweeps lie).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from torchlens._surface import render_surface_tsv, surface_rows

TABLE = Path(__file__).with_name("test_arch_spine_surface_table.tsv")


def _frozen_body() -> str:
    return "".join(
        line for line in TABLE.read_text().splitlines(keepends=True) if not line.startswith("#")
    )


def test_surface_lockstep_both_directions() -> None:
    live = render_surface_tsv(surface_rows())
    frozen = _frozen_body()
    assert live == frozen, (
        "root SURFACE drifted from the frozen table. If the facade change is "
        "deliberate, regenerate with `python tools/generate_surface.py` and "
        "review the diff (a NEW legacy_top_level=1 row is banned; see the L6+ "
        "ban test)."
    )


def test_every_lazy_name_resolves_in_a_fresh_interpreter() -> None:
    """Cold-resolution sweep: every facade row resolves at first touch."""

    code = (
        "import torchlens\n"
        "names = sorted(set(torchlens.__all__) | set(torchlens._LAZY_ATTRS))\n"
        "failures = []\n"
        "for name in names:\n"
        "    try:\n"
        "        getattr(torchlens, name)\n"
        "    except Exception as exc:\n"
        "        failures.append((name, repr(exc)))\n"
        "assert not failures, failures\n"
        "print(len(names))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, f"cold resolution failures:\n{result.stdout}\n{result.stderr}"
    assert int(result.stdout.strip()) >= 107
