#!/usr/bin/env python3
"""Snapshot and diff the attribute surface of ``torchlens.Trace``.

``--dump OUT.json`` records every name ``dir(Trace)`` yields with its kind and, for
callables and properties, its signature (the ``self`` annotation ignored). ``--diff
BASE.json HEAD.json`` compares two snapshots: public names (no leading underscore) that
were added, removed or re-signed fail the run; private differences are listed only.

Usage::

    python scripts/trace_surface_snapshot.py --dump outputs/head.json
    python scripts/trace_surface_snapshot.py --root ../base-checkout --dump outputs/base.json
    python scripts/trace_surface_snapshot.py --diff outputs/base.json outputs/head.json
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import sys
from pathlib import Path
from typing import Any


def _signature_text(obj: Any) -> str:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "<no signature>"
    params = []
    for index, param in enumerate(sig.parameters.values()):
        if index == 0 and param.name in ("self", "cls"):
            params.append(param.replace(annotation=inspect.Parameter.empty))
        else:
            params.append(param)
    return str(sig.replace(parameters=params))


def _describe(cls: type, name: str) -> dict[str, str]:
    raw = inspect.getattr_static(cls, name)
    if isinstance(raw, functools.cached_property):
        return {"kind": "cached_property", "signature": _signature_text(raw.func)}
    if isinstance(raw, property):
        parts = {"kind": "property", "signature": _signature_text(raw.fget) if raw.fget else ""}
        parts["setter"] = "yes" if raw.fset is not None else "no"
        return parts
    if isinstance(raw, staticmethod):
        return {"kind": "staticmethod", "signature": _signature_text(raw.__func__)}
    if isinstance(raw, classmethod):
        return {"kind": "classmethod", "signature": _signature_text(raw.__func__)}
    if inspect.isfunction(raw):
        return {"kind": "function", "signature": _signature_text(raw)}
    if inspect.isdatadescriptor(raw) or inspect.ismethoddescriptor(raw):
        return {"kind": type(raw).__name__, "signature": ""}
    return {"kind": "attribute:" + type(raw).__name__, "signature": ""}


def snapshot() -> dict[str, dict[str, str]]:
    from torchlens._data_classes.trace import Trace

    return {name: _describe(Trace, name) for name in dir(Trace)}


def diff(base: dict[str, dict[str, str]], head: dict[str, dict[str, str]]) -> int:
    names = sorted(set(base) | set(head))
    public_failures: list[str] = []
    private_notes: list[str] = []
    for name in names:
        if name.startswith("__") and name.endswith("__") and name not in base:
            continue
        before, after = base.get(name), head.get(name)
        if before == after:
            continue
        if before is None:
            line = f"ADDED    {name}: {after}"
        elif after is None:
            line = f"REMOVED  {name}: {before}"
        else:
            line = f"CHANGED  {name}: {before} -> {after}"
        public = not name.startswith("_") or (name.startswith("__") and name.endswith("__"))
        (public_failures if public else private_notes).append(line)
    for line in private_notes:
        print("private  " + line)
    for line in public_failures:
        print("PUBLIC   " + line)
    print(
        f"surface: {len(base)} base names, {len(head)} head names, "
        f"{len(public_failures)} public differences, {len(private_notes)} private differences"
    )
    return 1 if public_failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", help="checkout whose torchlens to import (default: this one)")
    parser.add_argument("--dump", metavar="OUT", help="write the snapshot JSON here")
    parser.add_argument("--diff", nargs=2, metavar=("BASE", "HEAD"), help="compare two snapshots")
    args = parser.parse_args(argv)
    if args.diff:
        base = json.loads(Path(args.diff[0]).read_text())
        head = json.loads(Path(args.diff[1]).read_text())
        return diff(base, head)
    if args.root:
        sys.path.insert(0, str(Path(args.root).resolve()))
    data = snapshot()
    out = Path(args.dump) if args.dump else None
    text = json.dumps(data, indent=1, sort_keys=True)
    if out is None:
        print(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"wrote {out} ({len(data)} names)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
