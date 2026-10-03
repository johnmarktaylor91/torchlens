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

import torchlens
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


def test_root_budget_is_frozen() -> None:
    """The root-name list is a FROZEN BUDGET (memo s5 namespace rule)."""

    rows = surface_rows()
    assert len(rows) == 152, (
        f"root surface budget moved: {len(rows)} names (frozen budget 152: "
        "143 at the C01 freeze, -1 private `_trace` alias retired, +6 by "
        "the A10 facade -- summary/load_extraction into __all__ plus the four "
        "pre-existing lazy namespaces bridge/callbacks/neuro/notebook -- "
        "+1 by C04: the lazy L4 submodule namespace tl.transforms (transforms "
        "memo s7 home; not in __all__) -- +2 by F29: the lazy submodule "
        "namespaces tl.agent (the agent inspection surface, FACADE, not in "
        "__all__) and tl.utils (the docs teach tl.utils.doctor(); the row was "
        "missing so the taught spelling raised on a cold import; L1, not in "
        "__all__) -- and +1 by F20: the lazy L6 submodule namespace "
        "tl.brainpipe (brainpipe memo s3 home, the memory-planned extraction "
        "planner; not in __all__, same class as tl.transforms). "
        "Root growth is a deliberate act: regenerate the table, justify the "
        "new name's layer, and update this pin in the same change."
    )
    in_all = [row for row in rows if row.in_all]
    assert len(in_all) == len(torchlens.__all__) == 116, (
        "__all__ budget moved -- update U-PUBLIC-OPERATIONS and this pin in "
        "the same deliberate change"
    )


def test_generated_all_matches_the_table() -> None:
    table_all = {row.canonical_path.split(".")[-1] for row in surface_rows() if row.in_all}
    assert table_all == set(torchlens.__all__)


def test_no_new_l6_plus_root_names() -> None:
    """Item 9: the ban on NEW L6+ root names is immediate."""

    grandfathered = {
        "torchlens.examples",
        "torchlens.export",
        "torchlens.facets",
        "torchlens.report",
        "torchlens.visualization",
        "torchlens.viz",
        # Pre-existing L6+/bridge namespaces the A10 facade made lazily
        # resolvable as root attributes (they were importable submodules long
        # before the freeze) -- namespaces, not new root verbs.
        "torchlens.bridge",
        "torchlens.callbacks",
        "torchlens.neuro",
        "torchlens.notebook",
    }
    live_legacy = {row.canonical_path for row in surface_rows() if row.legacy_top_level}
    new_legacy = live_legacy - grandfathered
    assert new_legacy == set(), (
        f"NEW L6+ root names: {sorted(new_legacy)} -- appliances never get root "
        "verbs (bare tl.NAME is a core-spine claim). Put the name under its "
        "subpackage namespace instead; membership in tl.<class>.* IS the "
        "peripheral badge."
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
    assert int(result.stdout.strip()) >= 112
