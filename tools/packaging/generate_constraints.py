"""Generate the packaging-test constraints files from ``tools/packaging/matrix.toml``.

Usage::

    python tools/packaging/generate_constraints.py           # write constraints/<set>.txt
    python tools/packaging/generate_constraints.py --check   # exit 1 when any file is stale
    python tools/packaging/generate_constraints.py --list-sets --group pr   # CI matrix JSON

Output is deterministic: sets and pins are sorted and no timestamp is written, so
a regeneration from an unchanged ``matrix.toml`` is byte-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # python 3.10: pytest depends on tomli there
    import tomli as tomllib  # type: ignore[no-redef]

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
MATRIX_PATH = HERE / "matrix.toml"
CONSTRAINTS_DIR = HERE / "constraints"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

EXPECTATIONS = ("install", "install_fails", "install_or_refuse")
GROUPS = ("pr", "extended")
HEADER = (
    "# Generated from tools/packaging/matrix.toml by "
    "tools/packaging/generate_constraints.py.\n"
    "# Do not edit: change matrix.toml and rerun the generator.\n"
)


class MatrixError(ValueError):
    """Raised when ``matrix.toml`` is inconsistent with itself or with pyproject."""


@dataclass(frozen=True)
class PackagingSet:
    """One named extras set of the packaging matrix."""

    name: str
    extras: tuple[str, ...]
    python: str
    expect: str
    group: str
    refuses: tuple[str, ...] = ()
    import_fails: tuple[str, ...] = ()
    present_gates: tuple[str, ...] = ()


def load_matrix(path: Path = MATRIX_PATH) -> dict[str, Any]:
    """Return the parsed matrix file."""

    with path.open("rb") as handle:
        return tomllib.load(handle)


def declared_extras(path: Path = PYPROJECT_PATH) -> set[str]:
    """Return the optional-dependency names pyproject.toml declares."""

    with path.open("rb") as handle:
        return set(tomllib.load(handle)["project"]["optional-dependencies"])


def resolve_sets(matrix: dict[str, Any], extras: set[str]) -> list[PackagingSet]:
    """Validate ``matrix`` against the declared ``extras`` and return its sets."""

    if matrix.get("schema") != 1:
        raise MatrixError(f"unsupported matrix schema {matrix.get('schema')!r}")
    default_python = matrix["defaults"]["python"]
    sets: list[PackagingSet] = []
    for name, raw in sorted(matrix["sets"].items()):
        entry = PackagingSet(
            name=name,
            extras=tuple(sorted(raw["extras"])),
            python=raw.get("python", default_python),
            expect=raw.get("expect", "install"),
            group=raw["group"],
            refuses=tuple(sorted(raw.get("refuses", ()))),
            import_fails=tuple(sorted(raw.get("import_fails", ()))),
            present_gates=tuple(sorted(raw.get("present_gates", ()))),
        )
        _validate_set(entry, extras, matrix.get("gates", {}))
        sets.append(entry)
    _validate_coverage(matrix, sets, extras)
    return sets


def _validate_set(entry: PackagingSet, extras: set[str], gates: dict[str, Any]) -> None:
    unknown = sorted(set(entry.extras) - extras)
    if unknown:
        raise MatrixError(f"set {entry.name!r} names undeclared extras {unknown}")
    if entry.expect not in EXPECTATIONS:
        raise MatrixError(f"set {entry.name!r} has unknown expect {entry.expect!r}")
    if entry.group not in GROUPS:
        raise MatrixError(f"set {entry.name!r} has unknown group {entry.group!r}")
    if set(entry.refuses + entry.present_gates) - set(gates):
        raise MatrixError(f"set {entry.name!r} names unknown gates")
    refusing = bool(entry.refuses or entry.import_fails)
    if (entry.expect == "install_or_refuse") != refusing:
        raise MatrixError(
            f"set {entry.name!r}: `refuses`/`import_fails` go with expect install_or_refuse"
        )


def _validate_coverage(matrix: dict[str, Any], sets: list[PackagingSet], extras: set[str]) -> None:
    aliases = matrix.get("aliases", {})
    for alias, target in aliases.items():
        if alias not in extras or target not in extras:
            raise MatrixError(f"alias {alias!r} -> {target!r} names an undeclared extra")
    singles = {entry.extras[0] for entry in sets if len(entry.extras) == 1}
    missing = sorted(extras - singles - set(aliases))
    if missing:
        raise MatrixError(f"extras with no single-extra set and no alias: {missing}")
    if not any(not entry.extras for entry in sets):
        raise MatrixError("the matrix needs a set with no extras")
    for gate_name, gate in matrix.get("gates", {}).items():
        if gate["extra"] not in extras:
            raise MatrixError(f"gate {gate_name!r} names undeclared extra {gate['extra']!r}")
    for extra in matrix["constraints"].get("extra", {}):
        if extra not in extras:
            raise MatrixError(f"constraints.extra names undeclared extra {extra!r}")


def render_constraints(matrix: dict[str, Any], entry: PackagingSet) -> str:
    """Return the constraints file text for one set."""

    pins = set(matrix["constraints"]["common"])
    per_extra = matrix["constraints"].get("extra", {})
    for extra in entry.extras:
        pins.update(per_extra.get(extra, ()))
    extras = ",".join(entry.extras) or "(none)"
    lines = [
        HEADER.rstrip("\n"),
        f"# set: {entry.name}; extras: {extras}; python: {entry.python}; expect: {entry.expect}",
        *sorted(pins),
    ]
    return "\n".join(lines) + "\n"


def expected_files(matrix: dict[str, Any], sets: list[PackagingSet]) -> dict[str, str]:
    """Return ``{file name: text}`` for every constraints file the matrix implies."""

    return {f"{entry.name}.txt": render_constraints(matrix, entry) for entry in sets}


def stale_files(expected: dict[str, str], directory: Path = CONSTRAINTS_DIR) -> list[str]:
    """Return the names of missing, differing or unexpected files in ``directory``."""

    stale = [
        name
        for name, text in expected.items()
        if not (directory / name).is_file() or (directory / name).read_text() != text
    ]
    if directory.is_dir():
        stale.extend(p.name for p in directory.glob("*.txt") if p.name not in expected)
    return sorted(stale)


def write_files(expected: dict[str, str], directory: Path = CONSTRAINTS_DIR) -> None:
    """Write every expected file and delete constraints files no set produces."""

    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.txt"):
        if path.name not in expected:
            path.unlink()
    for name, text in expected.items():
        (directory / name).write_text(text)


def ci_matrix(sets: list[PackagingSet], group: str) -> list[dict[str, str]]:
    """Return the GitHub Actions ``include`` list for ``group`` ("all" = every set)."""

    return [
        {"set": entry.name, "python": entry.python}
        for entry in sets
        if group == "all" or entry.group == group
    ]


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when a file is stale")
    parser.add_argument("--list-sets", action="store_true", help="print the CI matrix JSON")
    parser.add_argument("--group", default="all", choices=(*GROUPS, "all"))
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--pyproject", type=Path, default=PYPROJECT_PATH)
    parser.add_argument("--out", type=Path, default=CONSTRAINTS_DIR)
    args = parser.parse_args(argv)

    matrix = load_matrix(args.matrix)
    sets = resolve_sets(matrix, declared_extras(args.pyproject))
    if args.list_sets:
        print(json.dumps(ci_matrix(sets, args.group), sort_keys=True))
        return 0
    expected = expected_files(matrix, sets)
    if args.check:
        stale = stale_files(expected, args.out)
        for name in stale:
            print(f"stale: {args.out / name}", file=sys.stderr)
        if stale:
            print("run: python tools/packaging/generate_constraints.py", file=sys.stderr)
        return 1 if stale else 0
    write_files(expected, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
