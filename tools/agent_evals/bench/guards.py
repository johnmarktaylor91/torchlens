"""Import guards that disable ready-made implementations of a task's target method.

The primary track builds each method from primitives, so every arm disables any native
implementation of the target method. A guard entry is either a module (``pkg.sub``: importing it,
or anything under it, raises ``ImportError``) or an attribute (``pkg.sub:name`` or
``pkg.sub:Class.method``: the object is replaced by a stub that raises ``RuntimeError`` when
called, so the rest of the module keeps working; ``pkg.sub:*`` does this for every public
callable the module defines, for a module the library imports at package import).

The guard is installed into an arm environment as ``_agent_bench_guard.py`` plus a ``.pth`` line
that imports it at interpreter start. It reads the run's entries from the JSON file named by the
``AGENT_BENCH_GUARD`` variable. It fails closed: with the variable unset, or naming a missing or
unreadable file, it applies the union of every task's entries for the arm, written beside it at
install time. Unsetting the variable can only block more.

The guard deters; it does not police. ``audit_source`` scans the agent's code after the run for
any use of a guarded name, and a hit fails the run whether or not the guard fired.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

ENV_VAR = "AGENT_BENCH_GUARD"
MARKER = "disabled for this benchmark task"
MODULE_NAME = "_agent_bench_guard"
UNION_FILE = "_agent_bench_guard_all.json"
#: Harness-side natives per task and arm. Oracle authors never see the library arms, so the
#: entries live here, beside the arm definitions, not in the task manifests.
NATIVES_FILE = Path(__file__).resolve().parent / "natives.json"

GUARD_SOURCE = '''"""Benchmark import guard (installed by the agent benchmark harness)."""

import importlib.abc
import importlib.machinery
import json
import os
import sys

_MARKER = "disabled for this benchmark task"


def _entries():
    here = os.path.dirname(os.path.abspath(__file__))
    union = os.path.join(here, "_agent_bench_guard_all.json")
    path = os.environ.get("AGENT_BENCH_GUARD")
    for candidate in ([path] if path else []) + [union]:
        try:
            with open(candidate, encoding="utf-8") as handle:
                return list(json.load(handle)["blocked"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return []


def _stub(name):
    def disabled(*args, **kwargs):
        raise RuntimeError(name + " is " + _MARKER + "; build the method from primitives")

    disabled.__name__ = name.rsplit(".", 1)[-1].rsplit(":", 1)[-1]
    disabled._agent_bench_disabled = True
    return disabled


def _public_callables(module):
    return [
        name
        for name, obj in list(vars(module).items())
        if not name.startswith("_")
        and callable(obj)
        and getattr(obj, "__module__", None) == module.__name__
    ]


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, attrs):
        self._inner = inner
        self._attrs = attrs

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        for attr in self._attrs:
            if attr == "*":
                for name in _public_callables(module):
                    setattr(module, name, _stub(module.__name__ + ":" + name))
                continue
            owner = module
            parts = attr.split(".")
            for part in parts[:-1]:
                owner = getattr(owner, part, None)
                if owner is None:
                    break
            if owner is not None and hasattr(owner, parts[-1]):
                setattr(owner, parts[-1], _stub(module.__name__ + ":" + attr))


class _Guard(importlib.abc.MetaPathFinder):
    def __init__(self, entries):
        self.modules = [e for e in entries if ":" not in e]
        self.attrs = {}
        for entry in entries:
            if ":" in entry:
                mod, attr = entry.split(":", 1)
                self.attrs.setdefault(mod, []).append(attr)

    def find_spec(self, name, path, target=None):
        for mod in self.modules:
            if name == mod or name.startswith(mod + "."):
                raise ImportError(name + " is " + _MARKER + "; build the method from primitives")
        attrs = self.attrs.get(name)
        if not attrs:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader, attrs)
                return spec
        return None


_ENTRIES = _entries()
if _ENTRIES:
    sys.meta_path.insert(0, _Guard(_ENTRIES))
'''


def load_natives(path: Path = NATIVES_FILE) -> dict[str, dict[str, list[str]]]:
    """Return the harness's native entries, ``{task_id: {arm: [entry, ...]}}``."""

    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {task: dict(arms) for task, arms in raw.get("tasks", {}).items()}


def task_number(task_id: str) -> str:
    """Return the plan's short task name (``T1``) for a task directory id (``t01-jacobian-lens``)."""

    match = re.match(r"[tT]0*(\d+)", task_id)
    return f"T{match.group(1)}" if match else task_id


def entries_for(
    task_id: str,
    arm: str,
    manifest_natives: Mapping[str, Sequence[str]] | None = None,
    natives: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> list[str]:
    """Return the guard entries for one task in one arm (harness file plus task manifest).

    Entries listed under ``"*"`` apply to every arm.
    """

    table = load_natives() if natives is None else natives
    per_task = table.get(task_number(task_id), table.get(task_id, {}))
    found = [*per_task.get("*", []), *per_task.get(arm, [])]
    if manifest_natives:
        found += [*manifest_natives.get("*", []), *manifest_natives.get(arm, [])]
    return sorted(set(found))


def union_for(
    arm: str, natives: Mapping[str, Mapping[str, Sequence[str]]] | None = None
) -> list[str]:
    """Return every task's entries for ``arm``: what the guard applies when it fails closed."""

    table = load_natives() if natives is None else natives
    return sorted({entry for task in table for entry in entries_for(task, arm, natives=table)})


def install(site_packages: Path, union: Iterable[str]) -> list[Path]:
    """Write the guard module, its ``.pth`` hook and the fail-closed union into an environment.

    Returns
    -------
    list[Path]
        The files written.
    """

    module = site_packages / f"{MODULE_NAME}.py"
    pth = site_packages / f"{MODULE_NAME}.pth"
    union_path = site_packages / UNION_FILE
    module.write_text(GUARD_SOURCE, encoding="utf-8")
    pth.write_text(f"import {MODULE_NAME}\n", encoding="utf-8")
    union_path.write_text(json.dumps({"blocked": sorted(set(union))}, indent=2), encoding="utf-8")
    return [module, pth, union_path]


def write_run_config(path: Path, entries: Iterable[str]) -> Path:
    """Write one run's guard file (the file ``AGENT_BENCH_GUARD`` names)."""

    path.write_text(json.dumps({"blocked": sorted(set(entries))}, indent=2), encoding="utf-8")
    return path


PROBE = """
import importlib, json, sys
results = []
for entry in json.loads(sys.argv[1]):
    mod, _, attr = entry.partition(":")
    outcome, detail = "not blocked", ""
    try:
        obj = importlib.import_module(mod)
        if not attr:
            outcome = "not blocked"
        elif attr == "*":
            public = [n for n in dir(obj) if not n.startswith("_") and callable(getattr(obj, n))]
            stubbed = [n for n in public if getattr(getattr(obj, n), "_agent_bench_disabled", False)]
            own = [n for n in public if getattr(getattr(obj, n), "__module__", None) == mod]
            outcome = "blocked" if stubbed and not own else "not blocked"
            detail = str(len(stubbed)) + " callables disabled, " + str(len(own)) + " left"
        else:
            for part in attr.split("."):
                obj = getattr(obj, part)
            try:
                obj()
            except RuntimeError as exc:
                outcome, detail = ("blocked" if __MARKER__ in str(exc) else "not blocked"), str(exc)
            except Exception as exc:
                outcome, detail = "not blocked", type(exc).__name__ + ": " + str(exc)
    except ImportError as exc:
        outcome = "blocked" if __MARKER__ in str(exc) else "absent"
        detail = str(exc)
    except AttributeError as exc:
        outcome, detail = "absent", str(exc)
    results.append({"entry": entry, "outcome": outcome, "detail": detail[:300]})
print(json.dumps(results))
""".replace("__MARKER__", repr(MARKER))


def selftest(
    python: str, entries: Sequence[str], env: Mapping[str, str] | None = None
) -> list[dict[str, str]]:
    """Check in a fresh interpreter that each guard entry fires.

    Each result's ``outcome`` is ``blocked`` (the guard fired), ``absent`` (the target is not
    installed in this arm, so nothing to disable) or ``not blocked`` (a defect).
    """

    proc = subprocess.run(
        [python, "-c", PROBE, json.dumps(list(entries))],
        capture_output=True,
        text=True,
        env=dict(env) if env is not None else None,
        timeout=300,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"guard probe failed: {proc.stderr[-2000:]}")
    return list(json.loads(proc.stdout.strip().splitlines()[-1]))


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def audit_source(source: str, entries: Sequence[str]) -> list[str]:
    """Return every use of a guarded name in ``source`` (empty when clean or unparsable).

    Module entries match ``import``/``from`` statements of the module or below it. Attribute
    entries match ``from mod import name``, ``mod.name`` through any import alias of ``mod``,
    and ``getattr(obj, "name")`` with the attribute's last name (the usual bypass).
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    modules = [e for e in entries if ":" not in e]
    attrs = [tuple(e.split(":", 1)) for e in entries if ":" in e]
    hits: list[str] = []
    aliases: dict[str, str] = {}

    def under(name: str, mod: str) -> bool:
        return name == mod or name.startswith(mod + ".")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
                hits += [f"import {alias.name}" for m in modules if under(alias.name, m)]
        elif isinstance(node, ast.ImportFrom) and node.module:
            hits += [f"from {node.module} import" for m in modules if under(node.module, m)]
            for alias in node.names:
                full = f"{node.module}.{alias.name}"
                aliases[alias.asname or alias.name] = full
                hits += [
                    f"from {node.module} import {alias.name}" for m in modules if under(full, m)
                ]
                hits += [
                    f"from {node.module} import {alias.name}"
                    for mod, attr in attrs
                    if node.module == mod and (attr == "*" or alias.name == attr.split(".")[0])
                ]
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            if not dotted:
                continue
            head, _, rest = dotted.partition(".")
            resolved = f"{aliases.get(head, head)}.{rest}" if rest else aliases.get(head, head)
            for mod, attr in attrs:
                if (
                    resolved == f"{mod}.{attr}"
                    or resolved.endswith(f".{mod}.{attr}")
                    or attr == "*"
                    and resolved.startswith(f"{mod}.")
                    and resolved != mod
                ):
                    hits.append(dotted)
            hits += [dotted for m in modules if under(resolved, m)]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
        ):
            name = node.args[1].value
            hits += [f"getattr(..., {name!r})" for _, attr in attrs if name == attr.split(".")[-1]]
    return sorted(set(hits))
