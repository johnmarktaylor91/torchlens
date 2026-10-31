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
import pkgutil
import sys

from _harness import TESTS_DIR, run_checks

import torchlens
from torchlens._architecture import layer_for_module
from torchlens._surface import surface_rows

TABLE = TESTS_DIR / "test_arch_spine_surface_table.tsv"

#: Floor on the deep walk's denominator. Re-keyed 31 Oct 2026 to the decided
#: round-1 surface (lazy namespaces plus public non-lazy submodules, with
#: ``tl.debug``'s ``clone_input_tree`` and ``preserved_rng_state`` kept by
#: namespace ruling 1: 975 -> 977); it may only go down through a decided
#: surface change. 977 -> 976: the option receipt removal (e7449cd47) took
#: tl.options.OptionReceiptEntry off the surface. 976 -> 969: round 2 (module
#: moves) made the visualization (5 names) and semantic (2 names) packages
#: private. 969 -> 967: the 2.x saved-file cut removed ``tl.report.CompatWindow``
#: and ``tl.errors.ArtifactSchemaAgeWarning``.
DEEP_WALK_FLOOR = 967

_HEADER = (
    "canonical_path\thome_module\timplementation_layer\tproduct_role\tin_all\tlegacy_top_level"
)


def check_frozen_table_hygiene() -> None:
    """Sorted, duplicate-free, exact header: hand edits cannot hide."""

    lines = [line for line in TABLE.read_text().splitlines() if not line.startswith("#")]
    assert lines[0] == _HEADER, "frozen table header drifted from the wave-1 field set"
    paths = [line.split("\t")[0] for line in lines[1:] if line]
    assert paths == sorted(paths), "frozen table rows are not sorted by canonical_path"
    assert len(paths) == len(set(paths)), "frozen table holds duplicate canonical_path rows"


def check_dir_advertises_every_surface_row() -> None:
    """The five-step ``__getattr__``'s ``dir()`` covers the whole table."""

    listed = set(dir(torchlens))
    missing = sorted(
        row.canonical_path
        for row in surface_rows()
        if row.canonical_path.split(".")[-1] not in listed
    )
    assert missing == [], f"surface rows invisible to dir(torchlens): {missing}"


def check_deep_reachable_walk_resolves_with_layer_rows() -> None:
    """Every advertised member of every public namespace resolves to a layered home.

    The denominator is walked from code, never hand-listed: the root facade's
    lazy-attribute table names the lazy namespaces, ``pkgutil`` names the
    public non-lazy submodules, and each namespace's ``__all__`` names its
    advertised members (underscore names never count for the submodules). RED conditions: a member that raises on
    access, and a member whose torchlens home module resolves to no declared
    layer or role.
    """

    lazy = dict(torchlens._LAZY_ATTRS)
    unresolvable: list[tuple[str, str]] = []
    unlayered: list[tuple[str, str]] = []
    walked = 0
    advertised: list[tuple[str, object, list[str]]] = []
    lazy_paths = {module_path for module_path, attr in lazy.values() if attr is None}
    for _name, (module_path, attr) in sorted(lazy.items()):
        if attr is not None:
            continue  # plain name rows are covered by the root cold sweep
        namespace = importlib.import_module(module_path)
        advertised.append((module_path, namespace, list(getattr(namespace, "__all__", ()) or ())))
    # Public non-lazy submodules: only their public ``__all__`` names count.
    for info in sorted(pkgutil.iter_modules(torchlens.__path__), key=lambda i: i.name):
        module_path = f"torchlens.{info.name}"
        if info.name.startswith("_") or module_path in lazy_paths:
            continue
        namespace = importlib.import_module(module_path)
        names = [n for n in getattr(namespace, "__all__", ()) or () if not n.startswith("_")]
        advertised.append((module_path, namespace, names))
    for module_path, namespace, members in advertised:
        for member in members:
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
    print(f"\nDEEP SURFACE WALK: {walked} advertised members across public namespaces")
    assert walked >= DEEP_WALK_FLOOR, (
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


CHECKS = (
    check_frozen_table_hygiene,
    check_dir_advertises_every_surface_row,
    check_deep_reachable_walk_resolves_with_layer_rows,
)


def main() -> int:
    """Run every arch_tail_surface_audit check and return the exit status."""

    return run_checks("arch_tail_surface_audit", CHECKS)


if __name__ == "__main__":
    sys.exit(main())
