"""SURFACE-table audit tail: the deep reachable walk (F35; memo item 7).

C01's wave-1 lockstep (tests/test_arch_spine_surface.py) covers the ROOT
surface: frozen table byte-equality, budget pin, the L6+ ban, and the
fresh-interpreter cold sweep. This suite is the deep tail _surface.py
explicitly hands to lane F35: walk every public member ADVERTISED by every
lazy root namespace and demand the two RED conditions the memo names --
a reachable name that does not resolve, and a reachable name homed at a
module with no declared layer row ("a reachable name without a row ... or a
layer mismatch is RED").

Plus frozen-table hygiene: the committed TSV is sorted, duplicate-free, and
carries exactly the wave-1 header (so hand edits cannot masquerade as
generator output), and the root facade's ``dir()`` advertises every row.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import torchlens
from torchlens._architecture import layer_for_module
from torchlens._surface import surface_rows

TABLE = Path(__file__).with_name("test_arch_spine_surface_table.tsv")

_HEADER = (
    "canonical_path\thome_module\timplementation_layer\tproduct_role\tin_all\tlegacy_top_level"
)


@pytest.mark.smoke
def test_frozen_table_hygiene() -> None:
    """Sorted, duplicate-free, exact header: hand edits cannot hide."""

    lines = [line for line in TABLE.read_text().splitlines() if not line.startswith("#")]
    assert lines[0] == _HEADER, "frozen table header drifted from the wave-1 field set"
    paths = [line.split("\t")[0] for line in lines[1:] if line]
    assert paths == sorted(paths), "frozen table rows are not sorted by canonical_path"
    assert len(paths) == len(set(paths)), "frozen table holds duplicate canonical_path rows"


@pytest.mark.smoke
def test_dir_advertises_every_surface_row() -> None:
    """The five-step ``__getattr__``'s ``dir()`` covers the whole table."""

    listed = set(dir(torchlens))
    missing = sorted(
        row.canonical_path
        for row in surface_rows()
        if row.canonical_path.split(".")[-1] not in listed
    )
    assert missing == [], f"surface rows invisible to dir(torchlens): {missing}"


@pytest.mark.heavy
def test_deep_reachable_walk_resolves_with_layer_rows() -> None:
    """Every advertised member of every lazy namespace resolves to a layered home.

    The denominator is walked from code, never hand-listed: the root facade's
    lazy-attribute table names the namespaces; each namespace's ``__all__``
    names its advertised members. RED conditions: a member that raises on
    access, and a member whose torchlens home module resolves to no declared
    layer or role.
    """

    lazy = dict(torchlens._LAZY_ATTRS)
    unresolvable: list[tuple[str, str]] = []
    unlayered: list[tuple[str, str]] = []
    walked = 0
    for _name, (module_path, attr) in sorted(lazy.items()):
        if attr is not None:
            continue  # plain name rows are covered by the root cold sweep
        namespace = importlib.import_module(module_path)
        for member in getattr(namespace, "__all__", ()) or ():
            walked += 1
            try:
                value = getattr(namespace, member)
            except Exception as exc:  # noqa: BLE001 -- the audit reports, never masks
                unresolvable.append((f"{module_path}.{member}", repr(exc)))
                continue
            home = getattr(value, "__module__", None)
            if isinstance(home, str) and home.startswith("torchlens"):
                dotted = home[len("torchlens") :].lstrip(".")
                if layer_for_module(dotted) is None:
                    unlayered.append((f"{module_path}.{member}", home))
    print(f"\nDEEP SURFACE WALK: {walked} advertised members across lazy namespaces")
    assert walked >= 1000, (
        f"the deep walk shrank to {walked} members -- if namespaces stopped "
        "advertising __all__ this audit lost its denominator; investigate before "
        "adjusting the floor"
    )
    assert unresolvable == [], f"advertised members that do not resolve: {unresolvable}"
    assert unlayered == [], (
        f"advertised members homed at modules with no layer row: {unlayered} -- "
        "add the package/module row to torchlens/_architecture.py (or declare "
        "__tl_layer__ in the home module) in the same change that adds the name"
    )
