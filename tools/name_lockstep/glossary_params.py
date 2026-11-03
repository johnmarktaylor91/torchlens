"""Check that the glossary names every public parameter of every public callable.

The glossary (``docs/reference/glossary.md``) is the canonical API spec, so a caller must be
able to read there which keyword a callable takes (``Bundle.rank_changes(..., k=10)``, not
``top_k=``). This module lists every parameter a caller can pass by name and reports the ones
whose keyword row does not name the callable.

A parameter is public when a caller can pass it by name and its spelling has no leading
underscore: positional-or-keyword and keyword-only parameters, ``self`` and ``cls`` excepted.
Positional-only parameters (their name is not part of the call) and ``*args`` / ``**kwargs``
are not names a caller types, so they are out of scope.

A callable is public when the lockstep inventory reaches it (``inventory.build_inventory``):
functions and classes reachable from ``tl.*`` and its namespaces, and the public methods of
every TorchLens class those reach (classes count through their constructor).

The glossary names parameter ``p`` of callable ``C`` when a keyword row (a table row whose
first cell is `` `p=` ``) carries a backticked spelling of ``C``: its public path, any alias
the same object is reached by, or a dotted tail of one (`` `Bundle.rank_changes` `` names
``tl.Bundle.rank_changes``; `` `FireRecord` `` names ``tl.mechinterp.FireRecord``).
"""

from __future__ import annotations

import argparse
import enum
import importlib
import inspect
import json
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from name_lockstep import docsections, inventory

#: Parameter kinds a caller passes by name.
_NAMED_KINDS = frozenset({inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY})
_SKIPPED = frozenset({"self", "cls"})
#: A keyword row: first cell is exactly one backticked ``name=``.
_KEYWORD_ROW = re.compile(r"^\|\s*`([A-Za-z_][A-Za-z0-9_]*)=`\s*\|(.*)\|\s*$")
_SPAN = re.compile(r"`([^`]+)`")
#: The run of backticked owners right after ``keyword of`` (``A``, ``B`` and ``C``).
_OWNER_RUN = re.compile(r"keyword of ((?:`[^`]+`(?:,? and |, |,)?)+)")


def _member_target(row: dict[str, Any]) -> object | None:
    """Return the function a method member row names.

    Parameters
    ----------
    row
        A member row whose subkind ends in ``method``.

    Returns
    -------
    object | None
        The plain function (``classmethod`` / ``staticmethod`` unwrapped), or None.
    """

    owner: object = importlib.import_module(row["module"])
    for part in row["owner"].split(":", 1)[1].split("."):
        owner = getattr(owner, part, None)
    raw = vars(owner).get(row["name"]) if isinstance(owner, type) else None
    return raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw


def _member_spellings(row: dict[str, Any]) -> list[str]:
    """Return the ``Class.method`` spellings a member is reached by, public owners first.

    Parameters
    ----------
    row
        A member row.

    Returns
    -------
    list[str]
        ``Owner.name`` for every public reached owner, then the defining-class path.
    """

    public = [f"{owner}.{row['name']}" for owner in row["owner_names"] if not owner.startswith("_")]
    return sorted(set(public)) + ([row["path"]] if row["path"] not in public else [])


def public_callables(package: str = inventory.PACKAGE) -> list[tuple[list[str], object]]:
    """Return every public callable with every spelling it is reached by.

    Parameters
    ----------
    package
        Package to import.

    Returns
    -------
    list[tuple[list[str], object]]
        ``(spellings, callable)`` per distinct object; the first spelling is the display one.
    """

    root = importlib.import_module(package)
    objects = list(inventory._walk_objects(root))
    classes = {
        id(value): value
        for _, value in objects
        if inspect.isclass(value) and inventory._in_package(value.__module__, package)
    }
    found: dict[int, tuple[list[str], object]] = {}
    for row, value in objects:
        if row["subkind"] in ("function", "class"):
            found.setdefault(id(value), ([], value))[0].append(row["path"])
    for row in inventory._member_rows(classes.values(), package).values():
        if not row["subkind"].endswith("method") or row["name"].startswith("_"):
            continue
        target = _member_target(row)
        if target is not None:
            found.setdefault(id(target), ([], target))[0].extend(_member_spellings(row))
    return [(_ordered(paths), value) for paths, value in found.values()]


def _ordered(paths: Iterable[str]) -> list[str]:
    """Return unique spellings, public before underscored, then shortest first."""

    unique = set(paths)
    return sorted(unique, key=lambda p: (any(s.startswith("_") for s in p.split(".")), len(p), p))


def _inherited_from(cls: type, spellings: dict[int, list[str]]) -> list[str]:
    """Return the spellings of the public class whose constructor ``cls`` inherits unchanged.

    Parameters
    ----------
    cls
        A reached class.
    spellings
        Spellings of every reached callable, keyed on ``id``.

    Returns
    -------
    list[str]
        The defining base class's spellings, or an empty list when ``cls`` defines its own
        constructor, the base is not reached publicly, or the signatures differ.
    """

    base = next((k for k in cls.__mro__ if "__init__" in vars(k) or "__new__" in vars(k)), None)
    if base is None or base is cls or base is object or id(base) not in spellings:
        return []
    if str(inventory._signature(base)) != str(inventory._signature(cls)):
        return []
    return spellings[id(base)]


def _foreign_constructor(cls: type, package: str) -> bool:
    """Return whether a class's constructor signature is not TorchLens's own.

    Enum classes (their call signature is ``Enum``'s functional API) and classes whose
    constructor is defined outside ``package`` (a bare ``Exception`` or ``dict`` subclass)
    take no TorchLens keywords.
    """

    if issubclass(cls, enum.Enum):
        return True
    base = next((k for k in cls.__mro__ if "__init__" in vars(k) or "__new__" in vars(k)), None)
    return base is None or not inventory._in_package(base.__module__, package)


def _default_text(param: inspect.Parameter) -> str | None:
    """Return a short, stable spelling of a parameter's default, or None when it has none."""

    if param.default is param.empty:
        return None
    text = re.sub(r" at 0x[0-9a-f]+", "", repr(param.default))
    return text if len(text) <= 60 else "..."


def public_parameters(
    package: str = inventory.PACKAGE, *, meanings: bool = True
) -> list[dict[str, Any]]:
    """Return one row per public callable and public parameter.

    Parameters
    ----------
    package
        Package to import.
    meanings
        Read each parameter's documented meaning (slower; the coverage check needs none).

    Returns
    -------
    list[dict[str, Any]]
        Rows ``{callable, aliases, defined_by, param, kind, default, meaning}`` sorted by
        callable, param. ``defined_by`` is the spelling of the callable that defines the
        signature: the callable itself, or the public base class whose constructor a class
        inherits unchanged (its spellings join ``aliases``).
    """

    callables = public_callables(package)
    spellings = {id(value): paths for paths, value in callables}
    rows: list[dict[str, Any]] = []
    for paths, target in callables:
        if inspect.isclass(target) and _foreign_constructor(target, package):
            continue
        signature = inventory._signature(target)
        if signature is None:
            continue
        base = _inherited_from(target, spellings) if inspect.isclass(target) else []
        for param in signature.parameters.values():
            if param.kind not in _NAMED_KINDS or param.name in _SKIPPED:
                continue
            if param.name.startswith("_"):
                continue
            meaning = docsections.param_meaning(target, param.name)[0] if meanings else ""
            rows.append(
                {
                    "callable": paths[0],
                    "aliases": paths + base,
                    "defined_by": base[0] if base else paths[0],
                    "constructor": inspect.isclass(target),
                    "param": param.name,
                    "kind": param.kind.name.lower(),
                    "default": _default_text(param),
                    "meaning": meaning,
                }
            )
    return sorted(rows, key=lambda r: (r["callable"], r["param"]))


def keyword_rows(glossary: str) -> dict[str, list[str]]:
    """Return every keyword row of the glossary, keyed on keyword name.

    Parameters
    ----------
    glossary
        Glossary text.

    Returns
    -------
    dict[str, list[str]]
        Keyword name to the text after its first cell, one entry per row.
    """

    rows: dict[str, list[str]] = {}
    for line in glossary.splitlines():
        match = _KEYWORD_ROW.match(line.strip())
        if match:
            rows.setdefault(match.group(1), []).append(match.group(2))
    return rows


def _named_owners(text: str) -> set[str]:
    """Return the owners a keyword row names, with ``()`` and ``tl.`` stripped.

    Only the backticked run right after each ``keyword of`` counts, so a spelling that a
    row's meaning mentions in passing does not name an owner.
    """

    owners: set[str] = set()
    for run in _OWNER_RUN.findall(text):
        for span in _SPAN.findall(run):
            owners.add(span.strip().removesuffix("()").removeprefix("tl."))
    return owners


def _names(owner: str, aliases: Iterable[str]) -> bool:
    """Return whether glossary spelling ``owner`` names one of ``aliases``."""

    for alias in aliases:
        bare = alias.removeprefix("tl.")
        if bare == owner or bare.endswith(f".{owner}"):
            return True
    return False


def missing(rows: Iterable[dict[str, Any]], glossary: str) -> list[dict[str, Any]]:
    """Return the parameter rows the glossary does not name.

    Parameters
    ----------
    rows
        Rows from ``public_parameters``.
    glossary
        Glossary text.

    Returns
    -------
    list[dict[str, Any]]
        Rows with no keyword row naming their callable; each gains ``keyword_row`` (whether
        any row for the keyword exists).
    """

    table = keyword_rows(glossary)
    owners = {name: set().union(*map(_named_owners, texts)) for name, texts in table.items()}
    out: list[dict[str, Any]] = []
    for row in rows:
        named = owners.get(row["param"], set())
        if not any(_names(owner, row["aliases"]) for owner in named):
            out.append({**row, "keyword_row": row["param"] in table})
    return out


def _iter_report(rows: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> Iterator[str]:
    """Yield the summary lines the CLI prints."""

    callables = {r["callable"] for r in rows}
    yield f"public callables with named parameters: {len(callables)}"
    yield f"public (callable, parameter) pairs: {len(rows)}"
    yield f"pairs the glossary names: {len(rows) - len(gaps)}"
    no_row = sum(not g["keyword_row"] for g in gaps)
    yield f"pairs missing: {len(gaps)} ({no_row} whose keyword has no row at all)"


def main(argv: list[str] | None = None) -> int:
    """Dump public parameters and report glossary coverage; exit 1 when any is missing."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--glossary", type=Path, default=Path("docs/reference/glossary.md"))
    parser.add_argument("--dump", type=Path, help="write every public parameter row as JSON")
    parser.add_argument("--load", type=Path, help="read rows from a --dump file, skip import")
    args = parser.parse_args(argv)
    if args.load:
        rows = json.loads(args.load.read_text(encoding="utf-8"))
    else:
        rows = public_parameters(meanings=args.dump is not None)
    if args.dump:
        args.dump.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    gaps = missing(rows, args.glossary.read_text(encoding="utf-8"))
    for line in _iter_report(rows, gaps):
        print(line)
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
